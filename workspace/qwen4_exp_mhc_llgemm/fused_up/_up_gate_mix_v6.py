# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sixth fused mHC UP kernel variant: tcgen05, raw gn-kernels style.

Structured after gn-kernels sm100_mm_bf16.py and the in-tree
bf16x3_router_gemm_cutedsl.py: hand-built 128B-swizzled smem layouts, generic
cpasync.make_tiled_tma_atom, manual smem descriptors, raw _tcgen05 PTX
helpers (vllm.cute_utils._tcgen05) for MMA and tmem loads. No tiled_mma, no
make_tiled_tma_atom_A/B, no make_tmem_copy. Non-persistent grid, atom TMA +
descriptor prefetch (warp 1 prefetches before its PDL wait).

Same math and decomposition as v5 (_up_gate_mix_v5.py): swapped operands
(channels on tcgen05-M, tokens on N), one CTA per (64-channel, MMA_N-token)
tile, 4 stream accumulators packed 2-per-tmem-column-block into the M=64
stripe's idle half-lanes (stream s at lane offset 16*(s%2), column offset
MMA_N*(s//2) -- the make_fragment_C packing), deep weight TMA ring.

tmem layout note (MMA_M=64, 1-CTA): the hardware accumulator stripe layout
places channel row r at tmem lane (r%16) + 32*(r//16), leaving lanes 16-31
of each 32-lane group free; stream s%2==1 occupies those. The epilogue reads
full 32-lane rows (lane L holds channel ww*16+(L%16) for streams {L//16,
L//16+2}) and completes the 4-stream mix with a butterfly shuffle across
the half-warp, preserving the production accumulation order exactly.

Warp specialization (same split as v5, measured better than a single TMA
warp -- probe_v5_single_tma.py):
  warp 0  TMA weights (no PDL wait)
  warp 1  TMA descriptor prefetch, then PDL wait, then TMA activations
  warp 2  MMA (owns tmem alloc/dealloc)
  warps 4+  epilogue (raw tcgen05.ld -> sigmoid-mix -> st.global)
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync

from vllm.cute_utils import _tcgen05, simple_tma_copy


def _sigmoid_f32(x):
    return 1.0 / (1.0 + cute.math.exp(x * (-1.0)))


def _sigmoid_f32_tanh(x):
    # 1 MUFU.TANH (tanh.approx.f32) instead of MUFU.EX2 + MUFU.RCP.
    return 0.5 * cute.math.tanh(0.5 * x, approx=True) + 0.5


class UpGateMixV6Kernel:
    """Fused up-projection + sigmoid gate mix (tcgen05, raw/standard API).

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
        bk: int = 64,  # k-tile
        num_w_stage: int = 16,  # TMA ring depth for weights
        use_pdl: bool = False,
        sigmoid_mode: str = "tanh",  # "exact" (exp+rcp) or "tanh" (approx)
        num_epilog_wg: int = 1,  # epilogue warpgroups (128 threads each)
        epilog_ch: int = 0,  # xn/tmem chunk in tokens; 0 = auto
        gemm_only: bool = False,  # plain GEMM epilogue: store raw accs [M, N]
        debug_raw: int = -1,  # >=0: dump raw fp32 acc of this stream, no mix
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
        self.gemm_only = gemm_only
        self.debug_raw = debug_raw
        assert sigmoid_mode in ("exact", "tanh")
        assert num_epilog_wg in (1, 2, 4)
        assert num_epilog_wg == 1 or mma_n >= 8 * num_epilog_wg
        assert mma_m == 64  # epilogue lane mapping assumes the M=64 stripes
        assert hc % 2 == 0  # streams pack 2-per-tmem-column-block
        assert hc == 4  # epilogue mix order/shuffle pattern is HC=4-specific
        assert n % hc == 0 and self.hc_dim % mma_m == 0
        assert k % bk == 0 and bk % 16 == 0
        assert bk == 64  # 128B swizzle row; sdesc k-block advance hardcodes 32B
        assert mma_n % 8 == 0 and mma_n <= 128
        assert mma_n % num_epilog_wg == 0
        self.k_tiles = k // bk
        self.mma_k_blocks = bk // 16
        self.epilog_threads = 128 * num_epilog_wg
        self.threads_per_cta = 128 + self.epilog_threads
        # Full-NH xn prefetch when the register budget allows (<=384 epilogue
        # threads); chunk to 4 tokens otherwise (mma_n=64 -> 640 threads ->
        # 96-reg cap, full prefetch spills).
        nh = mma_n // num_epilog_wg
        self.epilog_ch = epilog_ch or (nh if self.epilog_threads <= 384 else 4)
        assert nh % self.epilog_ch == 0

    def _tmem_cols(self) -> int:
        # Streams are packed 2-per-column-block via the M=64 stripe's idle
        # half-lanes: stream s at lane offset 16*(s%2), col offset
        # MMA_N*(s//2) (same packing as make_fragment_C).
        cols = (self.hc // 2) * self.mma_n
        out = 32
        while out < cols:
            out *= 2
        return min(out, 512)

    @cute.jit
    def _make_tma(
        self,
        tensor: cute.Tensor,
        rows: cutlass.Constexpr,
        stages: cutlass.Constexpr,
    ):
        # K-major (rows, BK) tile, 128B swizzle, staged. Same construction as
        # gn-kernels _make_tma / bf16x3_router_gemm._make_tma.
        BK: cutlass.Constexpr = self.bk
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems: cutlass.Constexpr = 128 * 8 // tensor.element_type.width
        slayout = cute.make_layout(
            (rows, (elems, BK // elems), stages),
            stride=(elems, (1, rows * elems), rows * BK),
        )
        slayout = cute.make_composed_layout(swizzle_128B, 0, slayout)
        op = cpasync.CopyBulkTensorTileG2SOp()
        return cpasync.make_tiled_tma_atom(op, tensor, slayout, (rows, BK))

    @cute.jit
    def __call__(
        self,
        mW: cute.Tensor,  # [N, K] bf16, K-major (weights; GEMM A operand)
        mA: cute.Tensor,  # [M, K] bf16, K-major (activations; GEMM B operand)
        mXn: cute.Tensor,  # [M, N] bf16
        mOut: cute.Tensor,  # [M, HC_DIM] bf16
        stream,
    ):
        W_tma = self._make_tma(mW, self.mma_m, self.num_w_stage)
        A_tma = self._make_tma(mA, self.mma_n, self.k_tiles)

        smem_bytes = (
            (
                self.mma_m * self.bk * self.num_w_stage
                + self.mma_n * self.bk * self.k_tiles
            )
            * 2
            + (2 * self.num_w_stage + self.k_tiles + 2) * 8
            + 4
            + 2048  # alignment slack
        )
        self.kernel(W_tma, A_tma, mXn, mOut).launch(
            # grid.y = token tiles (M > mma_n spills to more CTAs; each CTA
            # re-reads the weights, shared via L2 across concurrent CTAs).
            grid=(
                self.hc_dim // self.mma_m,
                cute.ceil_div(cute.size(mXn, mode=[0]), self.mma_n),
                1,
            ),
            block=(self.threads_per_cta, 1, 1),
            smem=smem_bytes,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        W_tma: cpasync.TmaInfo,
        A_tma: cpasync.TmaInfo,
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

        M = cute.size(mXn, mode=[0])

        smem = cutlass.utils.SmemAllocator()
        sW = smem.allocate_tensor(
            BFloat16,
            W_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=W_tma.smem_layout.inner,
        )
        sA = smem.allocate_tensor(
            BFloat16,
            A_tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=A_tma.smem_layout.inner,
        )
        barW_full = smem.allocate_array(Int64, SW)
        barW_empty = smem.allocate_array(Int64, SW)
        barA_full = smem.allocate_array(Int64, KT)
        bar_mma_epilog = smem.allocate_array(Int64, 1)
        bar_tmem_alloc = smem.allocate_array(Int64, 1)
        tmem_base_ptr = smem.allocate_array(Int32, 1)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(SW):
                    cute.arch.mbarrier_init(barW_full + i, 1)
                    cute.arch.mbarrier_init(barW_empty + i, 1)
                for i in cutlass.range_constexpr(KT):
                    cute.arch.mbarrier_init(barA_full + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 32 + self.epilog_threads)
        elif warp_idx == 1:
            # Warm the TMA descriptor cache before the first loads; overlaps
            # the mbarrier init + the PDL wait below.
            cpasync.prefetch_descriptor(W_tma.atom)
            cpasync.prefetch_descriptor(A_tma.atom)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        w_stage_bytes = Int32(MMA_M * BK * 2)
        a_stage_bytes = Int32(MMA_N * BK * 2)

        if warp_idx == 0:
            # Weight TMA producer: no PDL wait -- weights are
            # producer-independent, so the cold-DRAM stream overlaps the
            # predecessor kernel.
            # (MMA_M, BK, RestM, RestK)
            gW_tiles = cute.local_tile(W_tma.tma_tensor, (MMA_M, BK), (None, None))
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
                simple_tma_copy(
                    W_tma.atom,
                    gW_tiles[None, None, tile_m, kt],
                    sW[None, None, stage],
                    barW_full + stage,
                )
                if stage == SW - 1:
                    empty_phase = empty_phase ^ 1
        elif warp_idx == 1:
            # Activation TMA producer: must wait for the predecessor kernel.
            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            # (MMA_N, BK, RestM, RestK); this CTA's token tile = bidm.
            gA_tiles = cute.local_tile(A_tma.tma_tensor, (MMA_N, BK), (None, None))
            for kt in cutlass.range(KT, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        barA_full + kt, a_stage_bytes
                    )
                simple_tma_copy(
                    A_tma.atom,
                    gA_tiles[None, None, bidm, kt],
                    sA[None, None, kt],
                    barA_full + kt,
                )
        elif warp_idx == 2:
            # MMA warp.
            num_tmem_cols = self._tmem_cols()
            if cutlass.const_expr(self.debug_raw != 201):
                cute.arch.alloc_tmem(num_tmem_cols, tmem_base_ptr, is_two_cta=False)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            if cutlass.const_expr(self.debug_raw != 201):
                cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
            tmem_base = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_base_ptr).toint()

            idesc = _tcgen05.make_bf16_idesc(MMA_M, MMA_N)
            sdesc = _tcgen05.make_sdesc_128B_swizzle(0)

            full_w_phase = cutlass.Int32(0)
            for kt in cutlass.range(KT, unroll=1):
                cute.arch.mbarrier_wait(barA_full + kt, 0)
                b_desc0 = sdesc | (sA[None, None, kt].iterator.toint() >> 4)
                for s in cutlass.range_constexpr(HC):
                    i = kt * HC + s
                    stage = i % SW
                    cute.arch.mbarrier_wait(barW_full + stage, full_w_phase)
                    a_desc = sdesc | (sW[None, None, stage].iterator.toint() >> 4)
                    b_desc = b_desc0
                    _tcgen05.fence_after_thread_sync()
                    # Stream s at lane offset 16*(s%2), col MMA_N*(s//2):
                    # packs 2 streams per column block into the M=64
                    # stripe's idle half-lanes (make_fragment_C packing).
                    d_tmem = tmem_base + (s % 2) * (16 << 16) + (s // 2) * MMA_N
                    if cutlass.const_expr(
                        self.debug_raw != 201
                        and (
                            self.debug_raw < 0
                            or self.debug_raw >= 10
                            or s == self.debug_raw
                        )
                    ):
                        for k_block in cutlass.range_constexpr(self.mma_k_blocks):
                            _tcgen05.mma_f16(
                                d_tmem,
                                a_desc,
                                b_desc,
                                idesc,
                                kt > 0 or k_block > 0,
                            )
                            a_desc += 32 >> 4
                            b_desc += 32 >> 4
                    _tcgen05.commit(barW_empty + stage)
                    if stage == SW - 1:
                        full_w_phase = full_w_phase ^ 1

            _tcgen05.commit(bar_mma_epilog)
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
            if cutlass.const_expr(self.debug_raw != 201):
                cute.arch.dealloc_tmem(
                    cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_base_ptr),
                    num_tmem_cols,
                    is_two_cta=False,
                )
        elif warp_idx >= 4:
            # Epilogue warps. NUM_EPILOG_WG warpgroups split the token
            # columns. With the interleaved stream packing (stream s at tmem
            # lane offset 16*(s%2), col block s//2), ALL lanes carry data:
            # lane L of warp ww holds channel ww*16+(L%16) for streams
            # {L//16, L//16+2}. The 4-stream mix is completed with a
            # butterfly shuffle across the half-warp (offset 16).
            epi_tid = tid - 128
            wg = epi_tid // 128
            tid128 = epi_tid % 128
            ww = tid128 // 32
            lane = tid128 % 32
            NWG: cutlass.Constexpr = self.num_epilog_wg
            NH: cutlass.Constexpr = MMA_N // NWG
            cute.arch.mbarrier_arrive(bar_tmem_alloc)
            cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
            tmem_base = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_base_ptr).toint()

            c0 = bid * MMA_M
            t0 = bidm * MMA_N
            s_lo = lane // 16  # this lane's streams: {s_lo, s_lo + 2}
            ch = c0 + ww * 16 + (lane % 16)
            lane_valid = lane < 16  # store predicate (mix result duplicated)
            CH: cutlass.Constexpr = self.epilog_ch

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # Prefetch xn chunk 0 into registers BEFORE waiting on the MMAs
            # (see v5 docstring); later chunks are double-buffered inside the
            # mix loop. Per lane: fixed channel, CH tokens x 2 streams per
            # chunk; consecutive same-half lanes cover consecutive channels,
            # so the 2B gathers coalesce in 32B segments. Chunking keeps the
            # live register set small at mma_n=64 (640 threads -> 96-reg cap;
            # full-NH scalar LDG destinations would exceed it).
            if cutlass.const_expr(not self.gemm_only and self.debug_raw < 0):
                rXn = [
                    [cute.make_rmem_tensor(CH, BFloat16) for _ in range(2)]
                    for _ in range(2)
                ]
                for j in cutlass.range_constexpr(2):
                    sc = s_lo + 2 * j
                    for tt in cutlass.range_constexpr(CH):
                        tok = t0 + wg * NH + tt
                        x = BFloat16(0.0)
                        if tok < M:
                            x = mXn[tok, sc * HC_DIM + ch]
                        rXn[0][j][tt] = x

            cute.arch.mbarrier_wait(bar_mma_epilog, 0)
            if cutlass.const_expr(self.debug_raw in (200, 201)):
                # debug: skip tmem load + mix; 200 keeps MMAs (MMA+TMA
                # floor), 201 skips them (pure TMA stream floor)
                cute.arch.mbarrier_arrive(bar_tmem_alloc)
            elif cutlass.const_expr(self.debug_raw >= 0):
                # debug: dump one stream's raw fp32 acc (correctness probes)
                s_dump = self.debug_raw % 10
                rGd = cute.make_rmem_tensor(NH, Float32)
                rGd.store(
                    _tcgen05.ld(
                        ww * 32,
                        tmem_base + (s_dump // 2) * MMA_N + wg * NH,
                        "32x32b",
                        NH,
                    )
                )
                _tcgen05.wait_ld()
                cute.arch.fence_view_async_tmem_load()
                cute.arch.mbarrier_arrive(bar_tmem_alloc)
                for t in cutlass.range_constexpr(NH):
                    tok = t0 + wg * NH + t
                    if tok < M and lane // 16 == s_dump % 2:
                        mOut[tok, ch] = rGd[t].to(BFloat16)
            elif cutlass.const_expr(self.gemm_only):
                # Plain GEMM epilogue: store the 4 stream accumulators to
                # mOut[M, N] in gate layout; no xn / sigmoid mix.
                for c in cutlass.range_constexpr(0, NH, CH):
                    rG = [cute.make_rmem_tensor(CH, Float32) for _ in range(2)]
                    for j in cutlass.range_constexpr(2):
                        col = tmem_base + j * MMA_N + wg * NH + c
                        rG[j].store(_tcgen05.ld(ww * 32, col, "32x32b", CH))
                    _tcgen05.wait_ld()
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(c + CH >= NH):
                        cute.arch.mbarrier_arrive(bar_tmem_alloc)
                    if cutlass.const_expr(c == 0):
                        if cutlass.const_expr(self.use_pdl):
                            cute.arch.griddepcontrol_launch_dependents()
                    for j in cutlass.range_constexpr(2):
                        sc = s_lo + 2 * j
                        for tt in cutlass.range_constexpr(CH):
                            tok = t0 + wg * NH + c + tt
                            if tok < M:
                                mOut[tok, sc * HC_DIM + ch] = rG[j][tt].to(BFloat16)
            else:
                for c in cutlass.range_constexpr(0, NH, CH):
                    cur: cutlass.Constexpr = (c // CH) % 2
                    nxt: cutlass.Constexpr = 1 - cur
                    if cutlass.const_expr(c + CH < NH):
                        # Prefetch next chunk's xn (overlaps this chunk's
                        # tmem load + mix).
                        for j in cutlass.range_constexpr(2):
                            sc = s_lo + 2 * j
                            for tt in cutlass.range_constexpr(CH):
                                tok = t0 + wg * NH + c + CH + tt
                                x = BFloat16(0.0)
                                if tok < M:
                                    x = mXn[tok, sc * HC_DIM + ch]
                                rXn[nxt][j][tt] = x
                    # Chunked tmem load: 2 col-blocks x CH token columns.
                    rG = [cute.make_rmem_tensor(CH, Float32) for _ in range(2)]
                    for j in cutlass.range_constexpr(2):
                        col = tmem_base + j * MMA_N + wg * NH + c
                        rG[j].store(_tcgen05.ld(ww * 32, col, "32x32b", CH))
                    _tcgen05.wait_ld()
                    # Make tcgen05.ld visible before TMEM release/RMEM use.
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(c + CH >= NH):
                        cute.arch.mbarrier_arrive(bar_tmem_alloc)
                    if cutlass.const_expr(c == 0):
                        if cutlass.const_expr(self.use_pdl):
                            cute.arch.griddepcontrol_launch_dependents()
                    # Production rounding boundary: fp32 gate -> bf16 -> fp32
                    # -> sigmoid; s-ordered serial mix; scale 1/HC; bf16 out.
                    # Lane L<16 holds streams {0,2}, L>=16 holds {1,3}; the
                    # butterfly shuffle fetches the other half's terms so the
                    # accumulation order ((s0+s1)+s2)+s3 is preserved exactly.
                    sig = cute.make_rmem_tensor((2, CH), Float32)
                    for j in cutlass.range_constexpr(2):
                        for tt in cutlass.range_constexpr(CH):
                            g = rG[j][tt].to(BFloat16).to(Float32)
                            if cutlass.const_expr(self.sigmoid_mode == "tanh"):
                                sig[j, tt] = _sigmoid_f32_tanh(g)
                            else:
                                sig[j, tt] = _sigmoid_f32(g)
                    for tt in cutlass.range_constexpr(CH):
                        term0 = sig[0, tt] * rXn[cur][0][tt].to(Float32)
                        term1 = sig[1, tt] * rXn[cur][1][tt].to(Float32)
                        otr0 = cute.arch.shuffle_sync_bfly(term0, 16)
                        otr1 = cute.arch.shuffle_sync_bfly(term1, 16)
                        tok = t0 + wg * NH + c + tt
                        if tok < M and lane_valid:
                            acc = ((term0 + otr0) + term1) + otr1
                            mOut[tok, ch] = (acc * (1.0 / HC)).to(BFloat16)
