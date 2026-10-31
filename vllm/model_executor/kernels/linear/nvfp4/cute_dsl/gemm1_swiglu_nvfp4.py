# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 gate_up GEMM + SwiGLU + NVFP4-quant epilogue (SM100/SM103).

Low-level CuTe-DSL kernel (hand-built TMA atoms and smem/instruction
descriptors, raw _tcgen05 ops, stage count from smem capacity) on an
INTERLEAVED gate_up weight: rows are permute-interleaved in 64-row blocks
[g64_0, u64_0, g64_1, u64_1, ...] (see interleave_gate_up_rows, applied at
weight-load time) and the 128x4-swizzled SFB covers those rows, so one 512B
swizzle atom holds the gate and up rows of one 64-column output block.
A BN=128 tile is one atom (cols [0,64) gate, [64,128) up); a BN=256 tile is
two contiguous atoms.

Configs (compile keys): (cta_group, BN, GROUP_M)
  - (2, 256, ...): m256n256 MMA, tmem acc [0,256)/[192,448) 64-col overlap,
    SFB multicast per rank. Large M.
  - (1, 128, ...): m128n128 MMA, tmem acc [0,128)/[128,256), no overlap.
    Small M: 2x the tiles of BN=256 for SM coverage.
GROUP_M = 0: flat (m-fastest) rasterization; 32/64: grouped, when B
exceeds half of L2.

SFB TMA: per rank loads its own 128-row block's stage (2KB) into slot
cta_rank, multicast to both CTAs (2-CTA config; plain copy for 1-CTA);
CTA0's MMA warp tcgen05.cp's both slots to tmem (4 cols per 128-row block
per k-block) and advances SF tmem pointers per k-block (no idesc SF-id
fields). Multicast credits the mapped barrier per destination CTA, so
stage tx counts SFB twice (see stage_size).

Epilogue math is packed f32x2 (TensorSSA vector arith on 32-wide halves);
MUFU exp2/rcp stay scalar. Output stores use L1::no_allocate (Q is consumed
by the next kernel, not reused on-chip). alpha/gsf_inv are scalar kernel
arguments: both are static per-layer constants (checkpoint scales fused at
weight load), so there is no per-call device-side scale computation.
"""

from dataclasses import dataclass

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import (
    Float4E2M1FN,
    Float32,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    cute,
)
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import make_fake_tensor
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.cute_utils import (
    EVICT_FIRST,
    _tcgen05,
    mbarrier,
    recast_val,
    simple_tma_copy,
    to_cta0_smem,
)
from vllm.cute_utils.cvt import fp32x2_to_fp8x2, fp32x8_to_fp4x8
from vllm.model_executor.warmup.jit_warmup_cutedsl_helper import VllmCuTeDSLJitKernel
from vllm.utils.math_utils import round_up
from vllm.utils.platform_utils import num_compute_units

_NEG_LOG2E = -1.4426950408889634
_SMEM_CAPACITY = cutlass.memory.get_smem_capacity_in_bytes("sm_100")


@dsl_user_op
def _e4m3_round(v: Float32, *, loc=None, ip=None) -> Uint32:
    """Round one fp32 to e4m3 (rn, satfinite); returns the byte in a u32."""
    return Uint32(fp32x2_to_fp8x2(v, v, loc=loc, ip=ip)) & Uint32(0xFF)


@dsl_user_op
def _e4m3_to_f32(b: Uint32, *, loc=None, ip=None) -> Float32:
    """Dequantize the e4m3 byte in the low bits of b to fp32."""
    out = llvm.inline_asm(
        T.f32(),
        [b.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo16, hi16;\n\t"
        ".reg .b16 flo, fhi;\n\t"
        ".reg .b32 f2;\n\t"
        "mov.b32 {lo16, hi16}, $1;\n\t"
        "cvt.rn.f16x2.e4m3x2 f2, lo16;\n\t"
        "mov.b32 {flo, fhi}, f2;\n\t"
        "cvt.f32.f16 $0, flo;\n\t"
        "}",
        "=f,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


class _Gemm1SwigluNvfp4:
    def __init__(self, cta_group: int, BN: int, GROUP_M: int) -> None:
        BM, BK = 128, 256
        self.cta_group = cta_group
        self.cta_tile = (BM, BN, BK)
        self.GROUP_M = GROUP_M
        # 2-CTA: SFB is multicast and credits the mapped barrier per
        # destination CTA (2x the box). 1-CTA: single plain copy.
        sfb_blocks = 2 if cta_group == 2 else 1
        # per-stage mbarrier tx bytes: A + this rank's B/SFB copy + SFA
        self.stage_size = (BM + BN // cta_group) * (BK // 2) + (
            BM // 128 + sfb_blocks
        ) * (BK // 64) * 512
        # per-stage smem: SFB reserves all cta_group slots
        smem_stage = (BM + BN // cta_group) * (BK // 2) + (BM // 128 + BN // 128) * (
            BK // 64
        ) * 512
        self.num_stages = _SMEM_CAPACITY // smem_stage

    @cute.jit
    def prepare_AB(self, A: cute.Tensor, BR: int, BK: int) -> cpasync.TmaInfo:
        tma_op = cpasync.CopyBulkTensorTileG2SOp(
            cta_group=tcgen05.CtaGroup.TWO
            if self.cta_group == 2
            else tcgen05.CtaGroup.ONE
        )
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        s_layout = cute.make_layout((BR, BK, self.num_stages), stride=(BK, 1, BR * BK))
        s_layout = cute.make_composed_layout(swizzle_128B, 0, s_layout)
        return cpasync.make_tiled_tma_atom(tma_op, A, s_layout, (BR, BK))

    @cute.jit
    def prepare_SF(
        self,
        SF: cute.Tensor,
        R: Int32,
        Ksf: Int32,
        BR: int,
        BKsf: int,
        multicast: bool,
    ) -> cpasync.TmaInfo:
        if cutlass.const_expr(multicast):
            tma_op = cpasync.CopyBulkTensorTileG2SMulticastOp(tcgen05.CtaGroup.TWO)
        else:
            tma_op = cpasync.CopyBulkTensorTileG2SOp(
                cta_group=tcgen05.CtaGroup.TWO
                if self.cta_group == 2
                else tcgen05.CtaGroup.ONE
            )
        # The swizzled 128x4 SF layout stores one opaque 512B atom per
        # (128 rows, 4 k-groups); treat the flat buffer as
        # (Ksf * 128, R // 128) and box-load one 128-row block's k-stage.
        g_layout = cute.make_layout((Ksf * 128, R // 128))
        SF_i64 = cute.recast_tensor(cute.make_tensor(SF.iterator, g_layout), Int64)
        tiler = (128 * BKsf // 8, BR // 128)
        s_layout = cute.make_layout((*tiler, self.num_stages))
        return cpasync.make_tiled_tma_atom(tma_op, SF_i64, s_layout, tiler)

    @cute.jit
    def __call__(
        self,
        A: cute.Tensor,  # (M_pad, K//2) uint8, packed e2m1
        B: cute.Tensor,  # (2I, K//2) uint8, interleaved 64-row blocks
        SFA: cute.Tensor,  # flat uint8, swizzled 128x4
        SFB: cute.Tensor,  # flat uint8, swizzled 128x4 over interleaved rows
        Q: cute.Tensor,  # (M, I//2) uint8 output
        SFQ: cute.Tensor,  # flat int32, swizzled output scales
        alpha: Float32,
        gsf_inv: Float32,
        grid_size: Int32,
        stream: CUstream,
    ) -> None:
        m_pad, k2 = A.shape
        n2 = B.shape[0]
        k = k2 * 2
        BM, BN, BK = self.cta_tile

        a_tensor = cute.make_tensor(
            cute.recast_ptr(A.iterator, dtype=Float4E2M1FN),
            cute.make_ordered_layout((m_pad, k, 1), order=(1, 0, 2)),
        )
        b_tensor = cute.make_tensor(
            cute.recast_ptr(B.iterator, dtype=Float4E2M1FN),
            cute.make_ordered_layout((n2, k, 1), order=(1, 0, 2)),
        )

        A_tma = self.prepare_AB(a_tensor, BM, BK)
        B_tma = self.prepare_AB(b_tensor, BN // self.cta_group, BK)
        # The swizzled SFA buffer always covers round_up(M_pad, 128) rows
        # (asserted in the wrapper); the TMA descriptor must claim at least
        # one 128-row block even when M_pad itself is smaller.
        SFA_tma = self.prepare_SF(
            SFA, round_up(m_pad, 128), k // 16, BM, BK // 16, False
        )
        SFB_tma = self.prepare_SF(
            SFB, n2, k // 16, BN // self.cta_group, BK // 16, self.cta_group == 2
        )

        self.kernel(A_tma, B_tma, SFA_tma, SFB_tma, Q, SFQ, alpha, gsf_inv).launch(
            grid=(grid_size, 1, 1),
            block=(6 * 32, 1, 1),
            cluster=(self.cta_group, 1, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        A_tma: cpasync.TmaInfo,
        B_tma: cpasync.TmaInfo,
        SFA_tma: cpasync.TmaInfo,
        SFB_tma: cpasync.TmaInfo,
        Q_tensor: cute.Tensor,
        SFQ_tensor: cute.Tensor,
        alpha: Float32,
        gsf_inv: Float32,
    ) -> None:
        tid, _, _ = cute.arch.thread_idx()
        raw_bid, _, _ = cute.arch.block_idx()
        num_bids, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        cta_group = self.cta_group
        cta_rank = raw_bid % cta_group

        BM, BN, BK = self.cta_tile
        GROUP_M = self.GROUP_M
        num_stages = self.num_stages

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(
            Float4E2M1FN,
            A_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=A_tma.smem_layout.inner,
        )
        sB = smem.allocate_tensor(
            Float4E2M1FN,
            B_tma.smem_layout.outer,
            byte_alignment=128,
            swizzle=B_tma.smem_layout.inner,
        )
        sSFA = smem.allocate_tensor(Int64, SFA_tma.smem_layout, byte_alignment=128)
        # SFB slots: rank r's (multicast) copy fills slot r; each slot is one
        # 128-row block's k-stage (BKsf * 128 bytes).
        BKsf = BK // 16
        sSFB = smem.allocate_tensor(
            Int64,
            cute.make_layout(
                (128 * BKsf // 8, cta_group, num_stages),
                stride=(1, 128 * BKsf // 8, (128 * BKsf // 8) * cta_group),
            ),
            byte_alignment=128,
        )

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)
        tmem_full_mbar = smem.allocate_array(Int64, 2)
        tmem_empty_mbar = smem.allocate_array(Int64, 2)
        if cutlass.const_expr(BN == 256):
            partial_mbar = smem.allocate_array(Int64, 1)
        taddr = smem.allocate(Int32, 4)

        BAR_EPI = 1
        BAR_TMEM_ALLOC = 2

        K = A_tma.tma_tensor.shape[1]
        N2 = B_tma.tma_tensor.shape[0]
        N = N2 // 2  # I (output columns)
        m_out = Q_tensor.shape[0]  # real tokens; A may be padded past this
        # 2-CTA tiles are counted in 128-row halves so each CTA owns one.
        grid_m = cute.ceil_div(m_out, BM * cta_group) * cta_group
        grid_n = N2 // BN
        total_tiles = grid_m * grid_n
        num_kt_out = N // 64  # 512B SF blocks per 128 output rows

        def compute_bid(bid, GROUP_M, grid_m, grid_n):
            # grid_m counts 128-row halves, so consecutive bids stay within
            # one CTA pair for the 2-CTA config.
            if cutlass.const_expr(GROUP_M == 0):
                bid_m = bid % grid_m
                bid_n = bid // grid_m
            else:
                group_size = GROUP_M * grid_n
                group_id = bid // group_size
                group_m = min(grid_m - group_id * GROUP_M, GROUP_M)
                local_bid = bid - group_id * group_size
                bid_m = group_id * GROUP_M + local_bid % group_m
                bid_n = local_bid // group_m
            return bid_m, bid_n

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_full_mbar + i, cta_group)
                    cute.arch.mbarrier_init(tma_empty_mbar + i, 1)
                for i in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_init(tmem_full_mbar + i, 1)
                    cute.arch.mbarrier_init(tmem_empty_mbar + i, 128 * cta_group)
                if cutlass.const_expr(BN == 256):
                    cute.arch.mbarrier_init(partial_mbar, 128 * cta_group)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(A_tma.atom)
            cpasync.prefetch_descriptor(B_tma.atom)
            cpasync.prefetch_descriptor(SFA_tma.atom)
            cpasync.prefetch_descriptor(SFB_tma.atom)

        cute.arch.cluster_arrive_relaxed()
        cute.arch.cluster_wait()

        cute.arch.griddepcontrol_wait()

        #
        # TMA load warp
        #
        if warp_id == 5:
            tma_stage = 0
            parity = 1
            tma_full_mbar_ = to_cta0_smem(tma_full_mbar)

            gA_tiles = cute.zipped_divide(A_tma.tma_tensor, (BM, BK))
            gB_tiles = cute.zipped_divide(B_tma.tma_tensor, (BN // cta_group, BK))
            gSFA_tiles = cute.zipped_divide(
                SFA_tma.tma_tensor, (128 * BKsf // 8, BM // 128)
            )
            gSFB_tiles = cute.zipped_divide(SFB_tma.tma_tensor, (128 * BKsf // 8, 1))

            for work_id in range(raw_bid, total_tiles, num_bids):
                bid_m, bid_n = compute_bid(work_id, GROUP_M, grid_m, grid_n)
                bid_n_ = bid_n * cta_group + cta_rank

                for iter_k in cutlass.range(K // BK, unroll=1):
                    mbar = tma_full_mbar_ + tma_stage
                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, parity)
                    with cute.arch.elect_one():
                        mbarrier.arrive_expect_tx(mbar, self.stage_size, "cluster")
                    # SFB first: for 2-CTA the 2-hop multicast delivery gates
                    # the stage barrier, so it must land under A/B.
                    if cutlass.const_expr(cta_group == 2):
                        simple_tma_copy(
                            SFB_tma.atom,
                            gSFB_tiles[None, (iter_k, bid_n_)],
                            sSFB[None, cta_rank, tma_stage],
                            mbar,
                            mcast_mask=Uint16(3),
                            cache_policy=EVICT_FIRST,
                        )
                    else:
                        simple_tma_copy(
                            SFB_tma.atom,
                            gSFB_tiles[None, (iter_k, bid_n)],
                            sSFB[None, 0, tma_stage],
                            mbar,
                            cache_policy=EVICT_FIRST,
                        )
                    simple_tma_copy(
                        A_tma.atom,
                        gA_tiles[None, (bid_m, iter_k, 0)],
                        sA[None, None, tma_stage],
                        mbar,
                    )
                    simple_tma_copy(
                        B_tma.atom,
                        gB_tiles[None, (bid_n_, iter_k, 0)],
                        sB[None, None, tma_stage],
                        mbar,
                        cache_policy=EVICT_FIRST,
                    )
                    simple_tma_copy(
                        SFA_tma.atom,
                        gSFA_tiles[None, (iter_k, bid_m)],
                        sSFA[None, None, tma_stage],
                        mbar,
                    )
                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        parity ^= 1

            cute.arch.griddepcontrol_launch_dependents()

        #
        # MMA warp (2-CTA collective; only cta_rank 0 issues)
        #
        if warp_id == 4:
            cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=2 * 32)

            if cta_rank == 0:
                tma_stage = 0
                tma_full_parity = 0
                tmem_stage = 0
                tmem_parity = 1

                MMA_M = BM * cta_group
                idesc = _tcgen05.make_nvfp4_idesc(MMA_M, BN)
                sdesc = _tcgen05.make_sdesc_128B_swizzle(0)
                # SF smem descriptor: no swizzle, SBO = 8 * 16 bytes.
                sdesc_sf = Uint64((8 << 32) | (1 << 46))
                mcast_mask = Uint16((1 << cta_group) - 1)

                # tmem map: acc stages then SF.
                #   BN=256: acc [0,256) / [192,448) (64-col overlap), SF [448,512)
                #   BN=128: acc [0,128) / [128,256),             SF [448,480)
                acc_stride = 192 if BN == 256 else 128
                sfa_tmem_base = 448
                sfb_cols_per_kb = 4 * (BN // 128)
                # SFB smem slot stride (one 128-row block's k-stage = 2KB) in
                # 16B descriptor units.
                SFB_SLOT_D = (128 * BKsf) >> 4

                for _work_id in range(raw_bid, total_tiles, num_bids):
                    acc_base = acc_stride * tmem_stage
                    cute.arch.mbarrier_wait(tmem_empty_mbar + tmem_stage, tmem_parity)
                    if cutlass.const_expr(BN == 256):
                        # wait for the previous tile's overlap-region read; the
                        # single partial barrier phases once per tile
                        cute.arch.mbarrier_wait(partial_mbar, tmem_stage ^ 1)
                    _tcgen05.fence_after_thread_sync()

                    for iter_k in cutlass.range(K // BK, unroll=1):
                        a_desc = sdesc | (
                            sA[None, None, tma_stage].iterator.toint() >> 4
                        )
                        b_desc = sdesc | (
                            sB[None, None, tma_stage].iterator.toint() >> 4
                        )
                        sfa_desc = sdesc_sf | (
                            sSFA[None, None, tma_stage].iterator.toint() >> 4
                        )
                        sfb_desc0 = sdesc_sf | (
                            sSFB[None, 0, tma_stage].iterator.toint() >> 4
                        )

                        cute.arch.mbarrier_wait(
                            tma_full_mbar + tma_stage, tma_full_parity
                        )
                        _tcgen05.fence_after_thread_sync()

                        # gn-style interleave: each k-block's SF copies
                        # immediately precede its MMA, hiding cp latency
                        # under MMA completion.
                        for k in cutlass.range_constexpr(BK // 64):
                            sfa_tmem = sfa_tmem_base + 4 * k
                            sfb_tmem = sfa_tmem_base + 16 + sfb_cols_per_kb * k
                            sf_off = 32 * k  # 512B per k-block, desc units
                            _tcgen05.cp(
                                sfa_tmem,
                                sfa_desc + sf_off,
                                "32x128b",
                                "warpx4",
                                cta_group,
                            )
                            for j in cutlass.range_constexpr(BN // 128):
                                _tcgen05.cp(
                                    sfb_tmem + 4 * j,
                                    sfb_desc0 + SFB_SLOT_D * j + sf_off,
                                    "32x128b",
                                    "warpx4",
                                    cta_group,
                                )

                            enable_input_d = iter_k > 0 or k > 0
                            _tcgen05.mma_nvfp4(
                                acc_base,
                                a_desc + 2 * k,
                                b_desc + 2 * k,
                                idesc,
                                sfa_tmem,
                                sfb_tmem,
                                enable_input_d,
                                cta_group,
                            )
                        _tcgen05.commit(
                            tma_empty_mbar + tma_stage, mcast_mask, cta_group
                        )

                        tma_stage = (tma_stage + 1) % num_stages
                        if tma_stage == 0:
                            tma_full_parity ^= 1

                    _tcgen05.commit(tmem_full_mbar + tmem_stage, mcast_mask, cta_group)
                    tmem_stage ^= 1
                    if tmem_stage == 0:
                        tmem_parity ^= 1

        #
        # Epilogue warps (tid < 128 indexes the tmem lane == output row)
        #
        if warp_id < 4:
            if warp_id == 0:
                _tcgen05.alloc(taddr, cta_group)
                cute.arch.barrier(barrier_id=BAR_TMEM_ALLOC, number_of_threads=2 * 32)

            rcp6 = Float32(1.0 / 6.0)
            rcp_gsf = cute.arch.rcp_approx(gsf_inv)

            # (1 row x 32B) tiles over the packed output: (32B), M, I//64
            mQt = cute.zipped_divide(Q_tensor, (1, 32))
            mQt = mQt[(0, None), None]

            acc_stride = 192 if BN == 256 else 128
            partial_ = to_cta0_smem(partial_mbar) if BN == 256 else None
            empty_ = to_cta0_smem(tmem_empty_mbar)

            tmem_stage = 0
            tmem_parity = 0
            lane_base = (tid // 32) * 32
            for work_id in range(raw_bid, total_tiles, num_bids):
                bid_m, bid_n = compute_bid(work_id, GROUP_M, grid_m, grid_n)
                row = bid_m * BM + tid
                acc_base = acc_stride * tmem_stage

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tmem_full_mbar + tmem_stage, tmem_parity)
                cute.arch.barrier(barrier_id=BAR_EPI, number_of_threads=4 * 32)
                _tcgen05.fence_after_thread_sync()

                # Gate/up column pairs in 64-col chunks: pair p covers output
                # cols [64*((BN//128)*bid_n + p), +64) with gate at
                # acc_base + 128p, up at +64. BN=256: the pair touching the
                # 64-col tmem overlap (abs cols [192,256)) is processed first
                # (stage 0 -> pair 1, stage 1 -> pair 0) and releases the
                # overlap after its loads; all order math is branch-free.
                for pi in cutlass.range_constexpr(BN // 128):
                    if cutlass.const_expr(BN == 256):
                        p = pi if tmem_stage == 1 else 1 - pi
                    else:
                        p = 0
                    g_col = acc_base + 128 * p

                    # one 128-col ld covers gate+up (contiguous in tmem);
                    # the (32, 4) view indexes the four 32-wide quarters
                    gu = _tcgen05.ld(lane_base, g_col, "32x32b", 128)
                    _tcgen05.wait_ld()
                    gu_q = gu.reshape((32, 4))

                    if cutlass.const_expr(BN == 256 and pi == 0):
                        _tcgen05.fence_before_thread_sync()
                        mbarrier.arrive(partial_, "cluster")
                    if cutlass.const_expr(pi == BN // 128 - 1):
                        _tcgen05.fence_before_thread_sync()
                        mbarrier.arrive(empty_ + tmem_stage, "cluster")

                    kb = (BN // 128) * bid_n + p
                    q_rmem = cute.make_rmem_tensor(8, Uint32)
                    sf_word = Uint32(0)
                    invv = cute.make_rmem_tensor(32, Float32)
                    for kh in cutlass.range_constexpr(2):
                        g = gu_q[None, kh] * alpha
                        e = cute.math.exp2(g * _NEG_LOG2E, fastmath=True)

                        dr = cute.make_rmem_tensor(32, Float32)
                        dr.store(e + 1.0)
                        for j in cutlass.range_constexpr(32):
                            dr[j] = cute.arch.rcp_approx(dr[j])

                        af = cute.make_rmem_tensor(32, Float32)
                        af.store(g * dr.load() * (gu_q[None, kh + 2] * alpha))

                        for kgloc in cutlass.range_constexpr(2):
                            amax0 = cute.math.abs(af[16 * kgloc])
                            amax1 = cute.math.abs(af[16 * kgloc + 1])
                            for j in cutlass.range_constexpr(7):
                                amax0 = cute.arch.fmax(
                                    amax0,
                                    cute.math.abs(af[16 * kgloc + 2 * j + 2]),
                                )
                                amax1 = cute.arch.fmax(
                                    amax1,
                                    cute.math.abs(af[16 * kgloc + 2 * j + 3]),
                                )
                            amax = cute.arch.fmax(amax0, amax1)
                            sfv = gsf_inv * (amax * rcp6)
                            sf_byte = _e4m3_round(sfv)
                            sfd = _e4m3_to_f32(sf_byte)
                            # Match csrc/quantization/fp4: when the rounded
                            # scale is zero, values scale by exactly 0.
                            inv = Float32(0.0)
                            if sfd != 0.0:
                                inv = cute.arch.rcp_approx(sfd * rcp_gsf)
                            for j in cutlass.range_constexpr(16):
                                invv[16 * kgloc + j] = inv
                            sf_word |= sf_byte << (8 * (2 * kh + kgloc))

                        af.store(af.load() * invv.load())

                        for kgloc in cutlass.range_constexpr(2):
                            kg = 2 * kh + kgloc
                            q_rmem[2 * kg] = fp32x8_to_fp4x8(af, 16 * kgloc)
                            q_rmem[2 * kg + 1] = fp32x8_to_fp4x8(af, 16 * kgloc + 8)

                    if row < m_out:
                        # Q is consumed by the next kernel (down GEMM); don't
                        # keep it in L1.
                        cute.arch.store(
                            cute.recast_ptr(
                                mQt[None, (row, kb)].iterator, dtype=Uint32
                            ),
                            q_rmem.load(),
                            level1_eviction_priority="evict_no_allocate",
                        )
                        sf_off = (
                            (row // 128 * num_kt_out + kb) * 128
                            + (row % 32) * 4
                            + (row // 32) % 4
                        )
                        SFQ_tensor[sf_off] = recast_val(sf_word, Int32)

                tmem_stage ^= 1
                if tmem_stage == 0:
                    tmem_parity ^= 1

            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
            if warp_id == 0:
                _tcgen05.dealloc(cta_group)


class Gemm1SwigluNvfp4Kernel(VllmCuTeDSLJitKernel["Gemm1SwigluNvfp4Kernel.CompileKey"]):
    """Compile-cached owner for the fused gate_up+SwiGLU+NVFP4 kernel."""

    @dataclass(frozen=True)
    class CompileKey:
        cta_group: int
        group_m: int

    def dispatch(  # type: ignore[override]
        self, *, m: int, n2: int, k2: int, num_sms: int
    ) -> "Gemm1SwigluNvfp4Kernel.CompileKey":
        # 1-CTA n128 while M fits one 128-row tile (decode); past that, 2-CTA
        # n256 when its wave still covers ~85% of the GPU (measured on B300:
        # cg2 wins from m=256 at 2I=34816, from m=1024 at 2I=8704).
        slots = num_sms // 2
        clusters = -(-m // 256) * (n2 // 256)
        waves = -(-clusters // slots)
        cta_group = 2 if m > 128 and clusters >= 0.85 * waves * slots else 1
        # Grouped rasterization keeps B panels L2-resident across a wave.
        # GROUP_M counts 128-row halves for 2-CTA (64 = 32 pair-tiles).
        group_m = 64 if cta_group == 2 else (32 if n2 * k2 > (63 << 20) else 0)
        return self.CompileKey(cta_group=cta_group, group_m=group_m)

    @staticmethod
    def kernel(compile_key: "Gemm1SwigluNvfp4Kernel.CompileKey"):
        cta_group = compile_key.cta_group
        return _Gemm1SwigluNvfp4(
            cta_group, 128 if cta_group == 1 else 256, compile_key.group_m
        )

    def warmup_inputs(self, compile_key: "Gemm1SwigluNvfp4Kernel.CompileKey"):
        m = cute.sym_int()
        m_pad = cute.sym_int()
        n2 = cute.sym_int(divisibility=256)

        A = make_fake_tensor(
            Uint8,
            (m_pad, cute.sym_int(divisibility=128)),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        B = make_fake_tensor(
            Uint8,
            (n2, cute.sym_int(divisibility=128)),
            (cute.sym_int64(divisibility=16), 1),
            assumed_align=16,
        )
        SFA = make_fake_tensor(
            Uint8, (cute.sym_int(divisibility=512),), (1,), assumed_align=16
        )
        SFB = make_fake_tensor(
            Uint8, (cute.sym_int(divisibility=512),), (1,), assumed_align=16
        )
        Q = make_fake_tensor(
            Uint8,
            (m, cute.sym_int(divisibility=64)),
            (cute.sym_int64(divisibility=64), 1),
            assumed_align=32,
        )
        SFQ = make_fake_tensor(
            Int32, (cute.sym_int(divisibility=128),), (1,), assumed_align=16
        )
        return (A, B, SFA, SFB, Q, SFQ, Float32(1.0), Float32(1.0), Int32(1))

    def get_warmup_keys(
        self, n2: int, k2: int, max_m: int
    ) -> list["Gemm1SwigluNvfp4Kernel.CompileKey"]:
        num_sms = num_compute_units()
        # m=1 reaches the 1-CTA config, max_m the 2-CTA one.
        return list(
            dict.fromkeys(
                self.dispatch(m=m, n2=n2, k2=k2, num_sms=num_sms) for m in (1, max_m)
            )
        )


_GEMM1_SWIGLU_NVFP4 = Gemm1SwigluNvfp4Kernel()


def gemm1_swiglu_nvfp4_supports(n2: int, k: int) -> bool:
    """Whether the kernel covers a gate_up of 2*I = n2 rows and K = k
    columns (per TP partition)."""
    return n2 % 256 == 0 and k % 256 == 0


def warmup_gemm1_swiglu_nvfp4(n2: int, k: int, max_m: int) -> None:
    """Register the compile keys reached for a gate_up of n2 rows and K = k
    columns, up to max_m tokens, for JIT warmup at init."""
    _GEMM1_SWIGLU_NVFP4.register_warmup(n2, k, max_m)


def gemm1_swiglu_nvfp4(
    a_fp4: torch.Tensor,
    a_sf: torch.Tensor,
    w_fp4: torch.Tensor,
    w_sf: torch.Tensor,
    alpha: float,
    out_global_scale_inv: float,
    m: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused gate_up GEMM + SwiGLU + NVFP4 quant.

    Args:
        a_fp4: Packed NVFP4 activations, [M_pad, K//2] uint8, as written by
            scaled_fp4_quant(x, gate_up.input_global_scale_inv,
            is_sf_swizzled_layout=True). Rows past ``m`` are read but never
            affect stored outputs; padding them to a multiple of 256 avoids
            TMA out-of-bounds fills at small M.
        a_sf: Activation block scales in the swizzled 128x4 layout, covering
            round_up(M_pad, 128) rows (rows >= m may hold any bytes).
        w_fp4: Packed gate_up weight, [2*I, K//2] uint8, rows interleaved in
            64-row [gate, up] blocks (see interleave_gate_up_rows; applied by
            the linear kernel at weight-load time).
        w_sf: Swizzled 128x4 weight scales over the interleaved rows.
        alpha: gate_up alpha, a static per-layer constant.
        out_global_scale_inv: down_proj's input_global_scale_inv, a static
            per-layer constant.
        m: Token count; defaults to a_fp4.shape[0].

    Returns:
        (q, sf): packed NVFP4 activations [M, I//2] uint8 and the swizzled
        scale tensor [round_up(M,128), round_up(I//16,4)//4] int32, byte
        compatible with torch.ops._C.silu_and_mul_nvfp4_quant's outputs.
        Scale rows >= M are left unwritten.

    """
    a_rows, k2 = a_fp4.shape
    m = a_rows if m is None else m
    n2 = w_fp4.shape[0]
    i = n2 // 2
    # The kernel indexes every operand flat; reject anything else here.
    assert a_fp4.dtype == torch.uint8 and a_fp4.is_contiguous()
    assert a_fp4.data_ptr() % 16 == 0
    assert w_fp4.dtype == torch.uint8 and w_fp4.is_contiguous()
    assert w_fp4.shape[1] == k2, f"K mismatch: A {k2 * 2}, W {w_fp4.shape[1] * 2}"
    assert gemm1_swiglu_nvfp4_supports(n2, k2 * 2)
    assert w_sf.is_contiguous() and w_sf.nbytes == n2 * k2 * 2 // 16
    assert a_sf.is_contiguous() and a_sf.element_size() in (1, 4)
    assert a_sf.nbytes >= round_up(a_rows, 128) * k2 * 2 // 16
    assert 0 <= m <= a_rows

    q = torch.empty((m, i // 2), dtype=torch.uint8, device=a_fp4.device)
    sf = torch.empty(
        (round_up(m, 128), round_up(i // 16, 4) // 4),
        dtype=torch.int32,
        device=a_fp4.device,
    )
    if m == 0:
        return q, sf

    num_sms = num_compute_units(a_fp4.get_device())
    key = _GEMM1_SWIGLU_NVFP4.dispatch(m=m, n2=n2, k2=k2, num_sms=num_sms)
    executor = _GEMM1_SWIGLU_NVFP4._get_or_compile(key)

    cta_group = key.cta_group
    grid_m = -(-m // (128 * cta_group)) * cta_group
    total = grid_m * (n2 // (128 if cta_group == 1 else 256))
    grid = min(total, num_sms // cta_group * cta_group)
    executor(
        a_fp4,
        w_fp4,
        a_sf.view(torch.uint8).view(-1),
        w_sf.view(torch.uint8).view(-1),
        q,
        sf.view(-1),
        alpha,
        out_global_scale_inv,
        grid,
    )
    return q, sf
