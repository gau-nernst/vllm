# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fifth fused mHC UP kernel variant: tcgen05 (SM100 tensor-memory MMA) with
swapped operands -- channels on the tcgen05-M dim, tokens on the N dim.

Same math as `_up_gate_mix.py`:

out[m, h] = (1/HC) * sum_s sigmoid(bf16(gate_s[m])) * xn[m, s*HC_DIM + h]
    gate_s[m] = dot(a[m, :], w[s*HC_DIM + h, :])   (fp32 accumulation)

Decomposition (GB300): one CTA per MMA_M output channels per MMA_N-token
tile (grid = (HC_DIM/MMA_M, ceil(M/MMA_N))). The CTA computes gate^T tiles
D[channel, token] = w[channels, :] @ a[tokens, :]^T with tcgen05.mma
(bf16 in, fp32 accum in TMEM), one accumulator per stream s
(HC=4 accumulators, each MMA_M x MMA_N fp32). Activations a[M, K] (tiny)
are TMA'd once per k-tile after the PDL wait; weights (6.5 MB total,
MMA_M*K*HC*2 B per CTA) are streamed through a deep TMA stage ring that
starts BEFORE griddepcontrol_wait (weights are producer-independent).
Token-tile CTAs re-read the weights but run concurrently, so DRAM sees the
stream once and the rest is L2 hits.

The epilogue tcgen05.ld's the 4 accumulators back with an identical
(channel, token) -> lane mapping, so the sigmoid-mix is purely per-lane:
round gate to bf16 (production rounding boundary), upcast, sigmoid,
multiply by xn[m, s*HC_DIM + h] (per-lane predicated gmem gather), sum
over s in production order, scale 1/HC, store bf16. The [M, N] gate tensor
is never materialized.

M is dynamic (token tiles cover any M; MMA_N <= 128 since the 4 stream
accumulators occupy 4*MMA_N of 512 tmem columns); the grid is static per
compiled bucket, so the kernel is CUDA-graph safe. Warp specialization
follows flashinfer's low-latency tcgen05 kernels (tgv_gemm_cute_ext.py /
dense_bf16_gemm_sm100_splitk.py):
  warp 0  TMA weights (no PDL wait)
  warp 1  TMA activations (griddepcontrol_wait first)
  warp 2  MMA (owns tmem alloc/dealloc)
  warps 4+  epilogue (tmem -> rmem -> sigmoid-mix -> st.global); NUM_EPILOG_WG
            warpgroups split the token columns (lane groups repeat per WG)
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05


def _sigmoid_f32(x):
    return 1.0 / (1.0 + cute.math.exp(x * (-1.0)))


def _sigmoid_f32_tanh(x):
    # 1 MUFU.TANH (tanh.approx.f32) instead of MUFU.EX2 + MUFU.RCP.
    return 0.5 * cute.math.tanh(0.5 * x, approx=True) + 0.5


class UpGateMixV5Kernel:
    """Fused up-projection + sigmoid gate mix (tcgen05, swapped operands).

    :compile-key: shapes (K, N, HC) static; M dynamic <= mma_n. One
        compilation per (mma_m, mma_n, bk, num_w_stage, use_pdl) config.
    """

    def __init__(
        self,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        mma_m: int = 64,  # channels per CTA per stream (tcgen05 M)
        mma_n: int = 32,  # padded tokens per tile (tcgen05 N), <= 128
        # (4 stream accumulators x mma_n tmem columns <= 512 tmem cols)
        bk: int = 64,  # k-tile
        num_w_stage: int = 8,  # TMA ring depth for weights
        use_pdl: bool = False,
        sigmoid_mode: str = "exact",  # "exact" (exp+rcp) or "tanh" (approx)
        num_epilog_wg: int = 1,  # epilogue warpgroups (128 threads each);
        # each takes an equal slice of the token columns
        debug_mode: int = 0,  # 0=full, 1=no phase-2 (gates STS only),
        # 2=MMA+TMA only (epilogue just hands tmem back)
        gemm_only: bool = False,  # plain GEMM epilogue: store 4 stream
        # accumulators to mOut[M, N], no xn prefetch / sigmoid mix
        single_tma_warp: bool = False,  # warp 0 issues weights AND
        # activations (ring fill -> PDL wait -> activation TMA -> rest);
        # requires SW <= HC*KT < 2*SW so empty phases are constant
    ):
        self.k = k
        self.n = n
        self.hc = hc
        self.hc_dim = n // hc
        self.mma_m = mma_m
        self.mma_n = mma_n
        self.bk = bk
        self.num_w_stage = num_w_stage
        self.use_pdl = use_pdl
        self.sigmoid_mode = sigmoid_mode
        self.num_epilog_wg = num_epilog_wg
        self.debug_mode = debug_mode
        self.gemm_only = gemm_only
        self.single_tma_warp = single_tma_warp
        if single_tma_warp:
            assert num_w_stage <= hc * (k // bk) < 2 * num_w_stage
        assert sigmoid_mode in ("exact", "tanh")
        assert num_epilog_wg in (1, 2, 4)
        assert num_epilog_wg == 1 or mma_n >= 8 * num_epilog_wg
        assert n % hc == 0 and self.hc_dim % mma_m == 0
        assert k % bk == 0 and bk % 16 == 0
        assert mma_n % 8 == 0 and mma_n <= 128
        assert mma_n % num_epilog_wg == 0
        self.k_tiles = k // bk
        self.mma_k_blocks = bk // 16
        self.epilog_threads = 128 * num_epilog_wg
        self.threads_per_cta = 128 + self.epilog_threads

    @cute.experimental.jit
    def __call__(
        self,
        mW: cute.Tensor,  # [N, K] bf16, K-major (weights; GEMM A operand)
        mA: cute.Tensor,  # [M, K] bf16, K-major (activations; GEMM B operand)
        mXn: cute.Tensor,  # [M, N] bf16
        mOut: cute.Tensor,  # [M, HC_DIM] bf16
        stream,
    ):
        self.kernel(mW, mA, mXn, mOut).launch(
            # grid.y = token tiles (M > mma_n spills to more CTAs; each CTA
            # re-reads the weights, shared via L2 across concurrent CTAs).
            grid=(
                self.hc_dim // self.mma_m,
                cute.ceil_div(cute.size(mA, mode=[0]), self.mma_n),
                1,
            ),
            block=(self.threads_per_cta, 1, 1),
            smem=Int64(utils.get_smem_capacity_in_bytes("sm_100")),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mW: cute.Tensor,
        mA: cute.Tensor,
        mXn: cute.Tensor,
        mOut: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid, bidm, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        MMA_M: cutlass.Constexpr = self.mma_m
        MMA_N: cutlass.Constexpr = self.mma_n
        BK: cutlass.Constexpr = self.bk
        HC: cutlass.Constexpr = self.hc
        HC_DIM: cutlass.Constexpr = self.hc_dim
        SW: cutlass.Constexpr = self.num_w_stage
        KT: cutlass.Constexpr = self.k_tiles
        mnk_tiler = (MMA_M, MMA_N, BK)
        cta_group = tcgen05.CtaGroup.ONE

        M = cute.size(mA, mode=[0])

        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            BFloat16,
            BFloat16,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            cta_group,
            (MMA_M, MMA_N),
        )

        sW = cute_ext.allocate(
            BFloat16,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, BFloat16, SW),
            alignment=1024,
        )
        sA = cute_ext.allocate(
            BFloat16,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, BFloat16, KT),
            alignment=1024,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, mnk_tiler, HC)

        barW_full = cute_ext.allocate(
            Int64, cute.AddressSpace.smem, cute.make_layout(SW), alignment=8
        ).iterator
        barW_empty = cute_ext.allocate(
            Int64, cute.AddressSpace.smem, cute.make_layout(SW), alignment=8
        ).iterator
        barA_full = cute_ext.allocate(
            Int64, cute.AddressSpace.smem, cute.make_layout(KT), alignment=8
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        ).iterator

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(SW):
                    cute.arch.mbarrier_init(barW_full + i, 1)
                    cute.arch.mbarrier_init(barW_empty + i, 1)
                for i in cutlass.range_constexpr(KT):
                    cute.arch.mbarrier_init(barA_full + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + self.epilog_threads)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        # Global tiles.
        # gW_t: (MMA_M, BK, N/MMA_M, K/BK); stream s channel tile = tile_m
        # s*(HC_DIM/MMA_M) + bid. gA_t: (MMA_N, BK, M-tiles, K/BK); this
        # CTA's token tile = bidm.
        gW_t = cute.local_tile(mW, (MMA_M, BK), (None, None))
        gA_t = cute.local_tile(mA, (MMA_N, BK), (bidm, None))
        vmap_w = cute_ext.get_cta_v_map_ab(mW, mnk_tiler, tiled_mma, "A")
        vmap_a = cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "B")

        w_stage_bytes = MMA_M * BK * 2
        a_stage_bytes = MMA_N * BK * 2

        if warp_idx == 0:
            # Weight TMA producer: no PDL wait -- weights are
            # producer-independent, so the cold-DRAM stream overlaps the
            # predecessor kernel.
            if cutlass.const_expr(self.single_tma_warp):
                # Single TMA warp: fill the weight ring, then (after the PDL
                # wait) the activations, then the remaining weight tiles.
                # Constant phases: the first SW waits pass on the initial
                # empty state (phase 1); the tail (HC*KT - SW < SW) waits on
                # phase 0.
                for i in cutlass.range(SW, unroll=1):
                    kt = i // HC
                    s = i % HC
                    tile_m = s * (HC_DIM // MMA_M) + bid
                    cute.arch.mbarrier_wait(barW_empty + i, cutlass.Int32(1))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barW_full + i, w_stage_bytes
                        )
                    cute_ext.tma_load(
                        gW_t[None, None, tile_m, kt],
                        sW[None, None, None, i],
                        (barW_full + i).value,
                        cta_v_map=vmap_w,
                        update_expect_tx=False,
                    )
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                for kt in cutlass.range(KT, unroll=1):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barA_full + kt, a_stage_bytes
                        )
                    cute_ext.tma_load(
                        gA_t[None, None, kt],
                        sA[None, None, None, kt],
                        (barA_full + kt).value,
                        cta_v_map=vmap_a,
                        update_expect_tx=False,
                    )
                for j in cutlass.range(HC * KT - SW, unroll=1):
                    i2 = j + SW
                    kt = i2 // HC
                    s = i2 % HC
                    tile_m = s * (HC_DIM // MMA_M) + bid
                    cute.arch.mbarrier_wait(barW_empty + j, cutlass.Int32(0))
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barW_full + j, w_stage_bytes
                        )
                    cute_ext.tma_load(
                        gW_t[None, None, tile_m, kt],
                        sW[None, None, None, j],
                        (barW_full + j).value,
                        cta_v_map=vmap_w,
                        update_expect_tx=False,
                    )
            else:
                empty_phase = cutlass.Int32(1)
                for i in cutlass.range(HC * KT, unroll=1):
                    kt = i // HC
                    s = i % HC
                    tile_m = s * (HC_DIM // MMA_M) + bid
                    stage = i % SW
                    cute.arch.mbarrier_wait(barW_empty + stage, empty_phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barW_full + stage, w_stage_bytes
                        )
                    cute_ext.tma_load(
                        gW_t[None, None, tile_m, kt],
                        sW[None, None, None, stage],
                        (barW_full + stage).value,
                        cta_v_map=vmap_w,
                        update_expect_tx=False,
                    )
                    if stage == SW - 1:
                        empty_phase = empty_phase ^ 1
        elif warp_idx == 1:
            # Activation TMA producer: must wait for the predecessor kernel.
            if cutlass.const_expr(not self.single_tma_warp):
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                for kt in cutlass.range(KT, unroll=1):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            barA_full + kt, a_stage_bytes
                        )
                    cute_ext.tma_load(
                        gA_t[None, None, kt],
                        sA[None, None, None, kt],
                        (barA_full + kt).value,
                        cta_v_map=vmap_a,
                        update_expect_tx=False,
                    )
        elif warp_idx == 2:
            # MMA warp.
            num_tmem_cols = self._tmem_cols()
            cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)

            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_base_ptr)
            acc_all = cute.make_tensor(tmem_ptr, acc_layout)
            mma_atom = cute.make_mma_atom(tiled_mma.op)

            full_w_phase = cutlass.Int32(0)
            for kt in cutlass.range(KT, unroll=1):
                cute.arch.mbarrier_wait(barA_full + kt, 0)
                for s in cutlass.range_constexpr(HC):
                    i = kt * HC + s
                    stage = i % SW
                    cute.arch.mbarrier_wait(barW_full + stage, full_w_phase)
                    if cutlass.const_expr(self.debug_mode != 3):
                        for k_block in cutlass.range_constexpr(self.mma_k_blocks):
                            mma_atom.set(
                                tcgen05.Field.ACCUMULATE, kt > 0 or k_block > 0
                            )
                            cute_ext.dot(
                                mma_atom,
                                cute.append_ones(
                                    sW[None, None, k_block, stage], up_to_rank=3
                                ),
                                cute.append_ones(
                                    sA[None, None, k_block, kt], up_to_rank=3
                                ),
                                acc_all[None, None, None, s],
                            )
                    with cute.arch.elect_one():
                        tcgen05.commit(barW_empty + stage, None, cta_group)
                    if stage == SW - 1:
                        full_w_phase = full_w_phase ^ 1

            with cute.arch.elect_one():
                tcgen05.commit(bar_mma_epilog, None, cta_group)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
            cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols, is_two_cta=False)
        elif warp_idx >= 4:
            # Epilogue warps. With num_epilog_wg=2, two warpgroups split the
            # token columns: warpgroup wg handles token columns
            # [wg*NH, (wg+1)*NH) of every stream accumulator (tcgen05.ld lane
            # groups repeat per warpgroup, so both can read the same tmem
            # rows at different column offsets).
            epi_tid = tid - 128
            wg = epi_tid // 128
            tid128 = epi_tid % 128
            NWG: cutlass.Constexpr = self.num_epilog_wg
            NH: cutlass.Constexpr = MMA_N // NWG
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)

            tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_base_ptr)
            acc_all = cute.make_tensor(tmem_ptr, acc_layout)

            acc_wg0 = cute.logical_divide(acc_all[((None, None), 0, 0, 0)], (None, NH))[
                None, (None, wg)
            ]
            tmem_load_op = sm100_utils.get_tmem_load_op(
                mnk_tiler,
                utils.LayoutEnum.ROW_MAJOR,
                BFloat16,
                Float32,
                (MMA_M, NH),
                False,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(tmem_load_op, acc_wg0)
            thr_t2r = tiled_copy_t2r.get_slice(tid128)

            epi_tile = (MMA_M, NH)
            cC = cute.make_identity_tensor(epi_tile)
            cC_epi = cute.flat_divide(cC, epi_tile)
            rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, cC_epi, tid128)
            rAcc = [
                cute_ext.allocate(
                    Float32, cute.AddressSpace.rmem, rmem_layout, alignment=32
                )
                for _ in range(HC)
            ]
            tCc = thr_t2r.partition_D(cC_epi[None, None, 0, 0])
            c0 = bid * MMA_M
            t0 = bidm * MMA_N
            VALS: cutlass.Constexpr = cute.size(rmem_layout)

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # Prefetch this thread's xn elements into registers BEFORE
            # waiting on the MMAs: xn is producer-dependent (hence after the
            # PDL wait) but independent of the gate MMAs, so the cold-DRAM
            # latency overlaps the weight stream / MMA pipeline instead of
            # extending it. The fragment layout is the tmem t2r layout
            # (channel, token), so these are scalar 2B gathers (coalesced
            # across lanes in 32B segments).
            rXn = None
            if cutlass.const_expr(self.debug_mode in (0, 4) and not self.gemm_only):
                rXn = [
                    cute_ext.allocate(
                        BFloat16,
                        cute.AddressSpace.rmem,
                        rmem_layout,
                        alignment=4,
                    )
                    for _ in range(HC)
                ]
                for s in cutlass.range_constexpr(HC):
                    for i in cutlass.range_constexpr(VALS):
                        ch = tCc[i][0]
                        tok = t0 + wg * NH + tCc[i][1]
                        x = BFloat16(0.0)
                        if tok < M:
                            x = mXn[tok, s * HC_DIM + c0 + ch]
                        rXn[s][i] = x

            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            for s in cutlass.range_constexpr(HC):
                acc_wg_s = cute.logical_divide(
                    acc_all[((None, None), 0, 0, s)], (None, NH)
                )[None, (None, wg)]
                cute_ext.partition_and_copy(thr_t2r, acc_wg_s, rAcc[s])
            # Make tcgen05.ld visible before TMEM release and RMEM use.
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(bar_tmem_alloc)

            if cutlass.const_expr(self.debug_mode >= 2):
                # MMA+TMA only: fold a value into out to defeat DCE.
                if epi_tid == 0 and bidm == 0:
                    mOut[0, bid * MMA_M] = rAcc[0][0].to(BFloat16)
            elif cutlass.const_expr(self.gemm_only):
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                # Plain GEMM epilogue: store the 4 stream accumulators to
                # mOut[M, N] in gate layout; no xn / sigmoid mix.
                for i in cutlass.range_constexpr(VALS):
                    ch = tCc[i][0]
                    tok = t0 + wg * NH + tCc[i][1]
                    if tok < M:
                        for s in cutlass.range_constexpr(HC):
                            mOut[tok, s * HC_DIM + c0 + ch] = rAcc[s][i].to(BFloat16)
            else:
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_launch_dependents()
                # Purely per-lane sigmoid gate mix: the 4 stream fragments
                # share the same (channel, token) -> lane mapping. Production
                # rounding boundary: fp32 gate -> bf16 -> fp32 -> sigmoid;
                # s-ordered serial mix; scale 1/HC; bf16 store.
                # Chunked two-phase mix: all HC sigmoids of a chunk are
                # independent MUFU chains (ILP) and land in distinct
                # registers; the s-ordered sum then fuses product+add
                # (FFMA) exactly like the production epilogue.
                CHUNK: cutlass.Constexpr = min(8, VALS)
                for i0 in cutlass.range_constexpr(0, VALS, CHUNK):
                    sig = cute.make_rmem_tensor((HC, CHUNK), Float32)
                    for s in cutlass.range_constexpr(HC):
                        for ii in cutlass.range_constexpr(CHUNK):
                            g = rAcc[s][i0 + ii].to(BFloat16).to(Float32)
                            if cutlass.const_expr(self.debug_mode == 4):
                                sig[s, ii] = g
                            elif cutlass.const_expr(self.sigmoid_mode == "tanh"):
                                sig[s, ii] = _sigmoid_f32_tanh(g)
                            else:
                                sig[s, ii] = _sigmoid_f32(g)
                    for ii in cutlass.range_constexpr(CHUNK):
                        ch = tCc[i0 + ii][0]
                        tok = t0 + wg * NH + tCc[i0 + ii][1]
                        if tok < M:
                            acc = Float32(0.0)
                            for s in cutlass.range_constexpr(HC):
                                acc += sig[s, ii] * rXn[s][i0 + ii].to(Float32)
                            mOut[tok, c0 + ch] = (acc * (1.0 / HC)).to(BFloat16)

    def _tmem_cols(self) -> int:
        cols = self.hc * self.mma_n
        out = 32
        while out < cols:
            out *= 2
        return min(out, 512)
