# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Third fused mHC UP kernel variant: activation-stationary, weight-streaming.

Same math as `_up_gate_mix.py`:

out[m, h] = (1/HC) * sum_s sigmoid(bf16(gate_s[m])) * xn[m, s*HC_DIM + h]
    gate_s[m] = dot(a[m, :], w[s*HC_DIM + h, :])   (fp32 accumulation)

Decomposition (GB300): one CTA per group of CH*CW output channels, one warp
per CW channels. Activations a[M, K] are read straight from gmem by each
warp (the 20KB working set sits in L1/L2, shared by all CTAs) -- a cp.async
SMEM staging pass measured SLOWER because its fill+barrier prologue costs
~1.5us and only pays off at large M. Each warp streams its channels' HC
weight rows into fp32 registers (lane = k-slice: vec8 main + vec2 tail, no
ragged predication) before the PDL wait, and accumulates fp32 partial dots
for an M-block of MB tokens in registers.

CW > 1 register-blocks over output channels: each lane loads an activation
K-fragment once per token and reuses it across CW*HC weight fragments, which
amortizes the bf16->fp32 activation converts (SHF) and the LDGs over CW x
more FFMAs and gives CW*HC*MB independent accumulator chains per lane. The
invariant CW*HC*MB = 32 keeps one staging row per lane, so the reduction and
epilogue structure (and the numerics: each dot is still 32 lane partials
reduced in warp_reduction_sum's butterfly tree order) are unchanged.

The cross-lane reduction goes through a bank-conflict-free padded SMEM
staging buffer (one butterfly-ordered tree sum per (channel, stream, token),
replicating warp_reduction_sum's order at lane 0). It is double-buffered and
runs ONE BLOCK BEHIND the dots (software pipeline): the scatter of block b is
issued right after its dots, while the gather + sigmoid-mix epilogue of block
b-1 overlaps the dots of block b. xn is likewise loaded one block ahead. The
final epilogue keeps the production s-order accumulation exactly. The [M, N]
gate tensor is never materialized.

Sub-warp dot segmentation (lanes_per_dot < 32, experimental): each dot is
covered by LPD lanes holding an interleaved K/LPD k-slice, so the warp runs
32/LPD independent dot groups side by side. This cuts the SMEM staging
traffic and the reduce tree ~4x at LPD=8, at the cost of 4x duplicated
activation loads/converts (the fragment cvt amortizes over CW*HC*LPD/32 rows
instead of CW*HC). Reduction order differs from LPD=32 (wider partials,
LPD-wide butterfly); bit-identity vs production is not guaranteed at LPD<32
(measured: still bit-identical at M<=2 on the shipped shape).

M is dynamic (<= max_m); grid is static, so the kernel is CUDA-graph safe.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr


def _sigmoid_f32(x):
    return 1.0 / (1.0 + cute.math.exp(x * (-1.0)))


class UpGateMixV3Kernel:
    """Fused up-projection + sigmoid gate mix (activation-stationary).

    :compile-key: shape-static except M; a single compilation covers all
        M <= max_m.
    """

    def __init__(
        self,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        max_m: int = 32,
        channels_per_cta: int = 4,
        channels_per_warp: int = 1,
        m_block: int = 8,
        m_splits: int = 1,
        lanes_per_dot: int = 32,
        fma2: int = 0,
        w256: int = 0,
        wna: int = 0,
        use_pdl: bool = False,
        debug_mode: int = 0,  # 0=full, 1=no-epilogue, 2=loads+dot only
        min_blocks_per_mp: int = 1,
    ):
        self.k = k
        self.n = n
        self.hc = hc
        self.hc_dim = n // hc
        self.max_m = max_m
        self.ch = channels_per_cta
        self.cw = channels_per_warp
        self.mb = m_block
        self.m_splits = m_splits
        self.lpd = lanes_per_dot
        self.fma2 = fma2
        self.w256 = w256
        self.wna = wna
        self.use_pdl = use_pdl
        self.debug_mode = debug_mode
        self.min_blocks_per_mp = min_blocks_per_mp
        self.num_warps = channels_per_cta  # one warp per CW channels
        self.num_threads = self.num_warps * cute.arch.WARP_SIZE
        assert n % hc == 0
        assert self.hc_dim % (channels_per_cta * channels_per_warp) == 0
        warp = cute.arch.WARP_SIZE
        if lanes_per_dot == warp:
            # K split: lane covers [lane*8, lane*8+8) and [256 + lane*2, +2).
            assert k == warp * 8 + warp * 2
        else:
            # Sub-warp dot: lane j covers vec8 k-groups {j + i*LPD} plus a
            # (K-256)/LPD-wide tail slice (interleaved LPD=32 generalization).
            assert warp % lanes_per_dot == 0
            assert (k - warp * 8) % lanes_per_dot == 0
        # Each dot group must own a whole number of (channel, stream) pairs.
        assert channels_per_warp * hc * lanes_per_dot % warp == 0
        if fma2:
            # Structural f32x2 pairs streams; needs the sub-warp path and an
            # even number of pairs per dot group.
            assert lanes_per_dot < warp
            assert channels_per_warp * hc * lanes_per_dot // warp % 2 == 0
        if w256:
            # 256-bit loads need a contiguous 32B main slice per lane:
            # 256/LPD elems, vec8-group aligned, and a vec-divisible tail.
            assert lanes_per_dot < warp
            assert not fma2
            assert (warp * 8) % lanes_per_dot == 0
            assert (256 // lanes_per_dot) % 8 == 0
            assert (k - warp * 8) % lanes_per_dot == 0
        # One staging row per lane: rows = CW * HC * MB = 32.
        assert channels_per_warp * hc * m_block == cute.arch.WARP_SIZE
        assert max_m % m_block == 0
        self.num_blocks = max_m // m_block

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # [M, K] bf16
        mW: cute.Tensor,  # [N, K] bf16
        mXn: cute.Tensor,  # [M, N] bf16
        mOut: cute.Tensor,  # [M, HC_DIM] bf16
        stream: CUstream,
    ):
        self.kernel(
            mA,
            mW,
            mXn,
            mOut,
            self.k,
            self.n,
            self.hc,
            self.hc_dim,
            self.max_m,
            self.ch,
            self.cw,
            self.mb,
            self.m_splits,
            self.num_blocks,
            self.num_threads,
            self.debug_mode,
            self.lpd,
            self.fma2,
            self.w256,
            self.wna,
        ).launch(
            # NOTE: making m_split the fastest-varying grid dim (same-channel
            # CTAs adjacent) measured WORSE at M<=4 (4.90/4.96 vs 4.58/4.64
            # us at M=1/4): at small M the phantom-exit CTAs interleave
            # between working ones in launch order and dilute residency.
            # Channel-group-fastest keeps all working (msub=0) CTAs dense.
            grid=[self.hc_dim // (self.ch * self.cw), self.m_splits, 1],
            block=[self.num_threads, 1, 1],
            smem=2 * self.ch * self.hc * self.mb * self.cw * (self.lpd + 1) * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    @cute.jit
    def _wload(self, src: cute.Tensor, dst: cute.Tensor, WNA: cutlass.Constexpr):
        # Weight-fragment load; WNA marks it L1 no-allocate (use-once
        # weights must not evict the reused activation/xn lines).
        if const_expr(WNA):
            cute.autovec_copy(
                src,
                dst,
                l1c_evict_priority=(cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE),
            )
        else:
            cute.autovec_copy(src, dst)

    @cute.jit
    def _dot_block(
        self,
        acc: cute.Tensor,  # (CW, HC, MB) fp32 rmem, zeroed here
        gA_vec: cute.Tensor,  # GMEM (M, (VEC, NUM_VEC)) bf16
        gA_v16: cute.Tensor,  # GMEM (M, (EMAIN, LPD)) bf16 (W256 only)
        gA_tail: cute.Tensor,  # GMEM (M, (2, WARP)) bf16
        wr: cute.Tensor,  # (CW, HC, VEC + 2) fp32 rmem weight fragments
        m0: cutlass.Int32,
        M: cutlass.Int32,
        lane: cutlass.Int32,
        CW: cutlass.Constexpr,
        HC: cutlass.Constexpr,
        MB: cutlass.Constexpr,
        VEC: cutlass.Constexpr,
        K: cutlass.Constexpr,
        LPD: cutlass.Constexpr,
        FMA2: cutlass.Constexpr,
        W256: cutlass.Constexpr,
    ):
        # Partial dots: lane's k-slice (8 + 2 elems) for MB tokens x CW
        # channels x HC streams. The a-fragment is loaded and widened ONCE
        # per token and reused across all CW*HC weight fragments (amortizes
        # the SHF converts and LDGs over CW x more FFMAs). a-slices come
        # straight from gmem (20KB working set, L1/L2 shared by all CTAs);
        # the row index is clamped so boundary blocks read a valid duplicate
        # row (garbage gates are discarded by the epilogue's store guard).
        # NOTE: packed f32x2 (FFMA2) does NOT dual-issue on SM103 -- the
        # FP32 rate is symmetric. Measured twice: the explicit
        # cute.arch.fma_packed_f32x2 intrinsic (dm2 @M=32: 5.38 vs 4.99us,
        # M<=4 4.80-4.90 vs 4.58-4.64us) and, on the sub-warp path, a
        # structural TensorSSA vector<2> pairing with zero packing MOVs
        # (SASS: 456 FFMA2, .F32x2.HI_LO pairs; dm2 @M=8 3.81 vs 3.74us,
        # M=1/4 4.90/4.93 vs 4.51/4.64us -- the fma2=1 flag keeps it for
        # reference). Scalar FFMA stays.
        acc.fill(0.0)
        if const_expr(LPD == cute.arch.WARP_SIZE):
            for ml in cutlass.range_constexpr(MB):
                m = min(m0 + ml, M - 1)
                a0 = cute.make_rmem_tensor_like(gA_vec[m, (None, lane)])
                cute.autovec_copy(gA_vec[m, (None, lane)], a0)
                a0_f32 = a0.load().to(cutlass.Float32)
                a1 = cute.make_rmem_tensor_like(gA_tail[m, (None, lane)])
                cute.autovec_copy(gA_tail[m, (None, lane)], a1)
                a1_f32 = a1.load().to(cutlass.Float32)
                for c in cutlass.range_constexpr(CW):
                    for s in cutlass.range_constexpr(HC):
                        acc_s = acc[c, s, ml]
                        for v in cutlass.range_constexpr(VEC):
                            acc_s += a0_f32[v] * wr[c, s, v]
                        for v in cutlass.range_constexpr(2):
                            acc_s += a1_f32[v] * wr[c, s, VEC + v]
                        acc[c, s, ml] = acc_s
        else:
            # Sub-warp dot: lane j of its LPD-lane group covers vec8 k-groups
            # {j + i*LPD} plus a TE-wide tail slice (interleaved split, the
            # generalization of the LPD=32 lane layout). The fragment convert
            # is amortized over PPG rows only (1:2 cvt:FFMA at LPD=8 vs 1:8
            # at LPD=32).
            PPG: cutlass.Constexpr = CW * HC * LPD // cute.arch.WARP_SIZE
            NV8: cutlass.Constexpr = cute.arch.WARP_SIZE // LPD
            TE: cutlass.Constexpr = (K - cute.arch.WARP_SIZE * VEC) // LPD
            j = lane % LPD
            for ml in cutlass.range_constexpr(MB):
                m = min(m0 + ml, M - 1)
                if const_expr(W256):
                    # Contiguous 32B main slice: one LDG.E.256 per token
                    # (lane j covers [j*EMAIN, +EMAIN) plus the tail).
                    EMAIN: cutlass.Constexpr = cute.arch.WARP_SIZE * VEC // LPD
                    am = cute.make_rmem_tensor_like(gA_v16[m, (None, j)])
                    cute.autovec_copy(gA_v16[m, (None, j)], am)
                    at = cute.make_rmem_tensor_like(gA_tail[m, (None, j)])
                    cute.autovec_copy(gA_tail[m, (None, j)], at)
                    am_f32 = am.load().to(cutlass.Float32)
                    at_f32 = at.load().to(cutlass.Float32)
                    for r in cutlass.range_constexpr(PPG):
                        acc_s = acc[r, ml]
                        for v in cutlass.range_constexpr(EMAIN):
                            acc_s += am_f32[v] * wr[r, v]
                        for v in cutlass.range_constexpr(TE):
                            acc_s += at_f32[v] * wr[r, EMAIN + v]
                        acc[r, ml] = acc_s
                    continue
                a_f32 = []
                for i in cutlass.range_constexpr(NV8):
                    ai = cute.make_rmem_tensor_like(gA_vec[m, (None, j + i * LPD)])
                    cute.autovec_copy(gA_vec[m, (None, j + i * LPD)], ai)
                    a_f32.append(ai.load().to(cutlass.Float32))
                at = cute.make_rmem_tensor_like(gA_tail[m, (None, j)])
                cute.autovec_copy(gA_tail[m, (None, j)], at)
                at_f32 = at.load().to(cutlass.Float32)
                if const_expr(FMA2):
                    # Structural f32x2: each stream pair accumulates as a
                    # vector<2> chain (mul.f32x2 + add.f32x2 -> FFMA2 in SASS
                    # without the packing MOVs the explicit fma_packed_f32x2
                    # intrinsic forced). Products of bf16-origin operands are
                    # exact in fp32, so the mul-first chain is bit-identical
                    # to the scalar fma chain.
                    ELE: cutlass.Constexpr = NV8 * VEC + TE
                    ab = []
                    for i in cutlass.range_constexpr(NV8):
                        for v in cutlass.range_constexpr(VEC):
                            ab.append(cute.full((2,), a_f32[i][v], cutlass.Float32))
                    for v in cutlass.range_constexpr(TE):
                        ab.append(cute.full((2,), at_f32[v], cutlass.Float32))
                    for rp in cutlass.range_constexpr(PPG // 2):
                        accp = wr[rp, (None, 0)].load() * ab[0]
                        for v in cutlass.range_constexpr(1, ELE):
                            accp = accp + wr[rp, (None, v)].load() * ab[v]
                        acc[2 * rp, ml] = accp[0]
                        acc[2 * rp + 1, ml] = accp[1]
                else:
                    for r in cutlass.range_constexpr(PPG):
                        acc_s = acc[r, ml]
                        for i in cutlass.range_constexpr(NV8):
                            for v in cutlass.range_constexpr(VEC):
                                acc_s += a_f32[i][v] * wr[r, i * VEC + v]
                        for v in cutlass.range_constexpr(TE):
                            acc_s += at_f32[v] * wr[r, NV8 * VEC + v]
                        acc[r, ml] = acc_s

    @cute.jit
    def _scatter(
        self,
        acc: cute.Tensor,  # (CW, HC, MB) fp32 rmem
        sm_stage: cute.Tensor,  # SMEM (2, CH, CW*HC*MB, CW*HC*MB+1) fp32
        mOut: cute.Tensor,
        m0: cutlass.Int32,
        par: cutlass.Int32,
        gw: cutlass.Int32,  # global warp id (channel group)
        M: cutlass.Int32,
        wid: cutlass.Int32,
        lane: cutlass.Int32,
        CW: cutlass.Constexpr,
        HC: cutlass.Constexpr,
        MB: cutlass.Constexpr,
        LPD: cutlass.Constexpr,
        DEBUG_MODE: cutlass.Constexpr,
    ):
        if const_expr(DEBUG_MODE == 2):
            # Loads + dot only: fold partials into out to defeat DCE.
            s_l = (lane % (HC * MB)) // MB
            if s_l == 0:
                m = m0 + lane % MB
                if m < M:
                    if const_expr(LPD == cute.arch.WARP_SIZE):
                        v = acc[0, 0, lane % MB]
                    else:
                        v = acc[0, lane % MB]
                    mOut[m, gw * CW + lane // (HC * MB)] = v.to(cutlass.BFloat16)
            return
        # Scatter partials; row (pair*MB + ml) holds LPD lane partials.
        cute.arch.sync_warp()
        if const_expr(LPD == cute.arch.WARP_SIZE):
            for c in cutlass.range_constexpr(CW):
                for s in cutlass.range_constexpr(HC):
                    for ml in cutlass.range_constexpr(MB):
                        sm_stage[par, wid, c * (HC * MB) + s * MB + ml, lane] = acc[
                            c, s, ml
                        ]
        else:
            PPG: cutlass.Constexpr = CW * HC * LPD // cute.arch.WARP_SIZE
            g = lane // LPD
            j = lane % LPD
            for r in cutlass.range_constexpr(PPG):
                for ml in cutlass.range_constexpr(MB):
                    sm_stage[par, wid, (PPG * g + r) * MB + ml, j] = acc[r, ml]
        cute.arch.sync_warp()

    @cute.jit
    def _reduce_epilogue(
        self,
        sm_stage: cute.Tensor,  # SMEM (2, CH, CW*HC*MB, CW*HC*MB+1) fp32
        mOut: cute.Tensor,
        m0: cutlass.Int32,
        xn_b: cutlass.Float32,
        par: cutlass.Int32,
        gw: cutlass.Int32,  # global warp id (channel group)
        M: cutlass.Int32,
        wid: cutlass.Int32,
        lane: cutlass.Int32,
        CW: cutlass.Constexpr,
        HC: cutlass.Constexpr,
        MB: cutlass.Constexpr,
        WARP: cutlass.Constexpr,
        LPD: cutlass.Constexpr,
        DEBUG_MODE: cutlass.Constexpr,
    ):
        # Lane l reduces row l over the LPD lane partials, replicating
        # warp_reduction_sum's butterfly order (LPD-wide tree at LPD < 32).
        vals = cute.make_rmem_tensor((LPD,), cutlass.Float32)
        for i in cutlass.range_constexpr(LPD):
            vals[i] = sm_stage[par, wid, lane, i]
        NSTAGE: cutlass.Constexpr = LPD.bit_length() - 1
        for off in cutlass.range_constexpr(NSTAGE):
            step: cutlass.Constexpr = (LPD // 2) >> off
            for i in cutlass.range_constexpr(step):
                vals[i] = vals[i] + vals[i + step]

        # Lane l owns (channel c_l, stream s_l, token offset ml_l).
        c_l = lane // (HC * MB)
        s_l = (lane % (HC * MB)) // MB
        ml_l = lane % MB
        h = gw * CW + c_l
        if const_expr(DEBUG_MODE == 1):
            if s_l == 0:
                m = m0 + ml_l
                if m < M:
                    mOut[m, h] = vals[0].to(cutlass.BFloat16)
        else:
            # Production rounding boundary: fp32 gate -> bf16 -> fp32 ->
            # sigmoid, then mix with the prefetched xn.
            g = vals[0].to(cutlass.BFloat16).to(cutlass.Float32)
            contrib = _sigmoid_f32(g) * xn_b
            # Gather the HC stream contributions for this lane's (channel,
            # token) in stream order (serial sum == production epilogue).
            out_acc = cutlass.Float32(0.0)
            for s in cutlass.range_constexpr(HC):
                c = cute.arch.shuffle_sync(contrib, c_l * (HC * MB) + s * MB + ml_l)
                out_acc += c
            if s_l == 0:
                m = m0 + ml_l
                if m < M:
                    mOut[m, h] = (out_acc * (1.0 / HC)).to(cutlass.BFloat16)

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mW: cute.Tensor,
        mXn: cute.Tensor,
        mOut: cute.Tensor,
        K: cutlass.Constexpr,
        N: cutlass.Constexpr,
        HC: cutlass.Constexpr,
        HC_DIM: cutlass.Constexpr,
        MAX_M: cutlass.Constexpr,
        CH: cutlass.Constexpr,
        CW: cutlass.Constexpr,
        MB: cutlass.Constexpr,
        M_SPLITS: cutlass.Constexpr,
        NUM_BLOCKS: cutlass.Constexpr,
        NUM_THREADS: cutlass.Constexpr,
        DEBUG_MODE: cutlass.Constexpr,
        LPD: cutlass.Constexpr,
        FMA2: cutlass.Constexpr,
        W256: cutlass.Constexpr,
        WNA: cutlass.Constexpr,
    ):
        _, _, _ = cute.arch.thread_idx()
        h0, msub, _ = cute.arch.block_idx()
        wid = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        WARP: cutlass.Constexpr = cute.arch.WARP_SIZE
        VEC: cutlass.Constexpr = 8

        M = cute.size(mA, mode=[0])
        gw = h0 * CH + wid  # this warp's channel group; channels gw*CW + c

        # Fire at CTA start: dependents may launch immediately and hide their
        # startup under our weight stream + dots. Pure launch hint -- the
        # consumer's griddepcontrol_wait still waits for our full completion.
        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # Phantom CTAs (their first M-block is past M) exit before streaming
        # any weights -- they would otherwise double the cold w DRAM traffic
        if msub * MB < M:  # phantom CTAs skip all weight traffic
            # (rows, (VEC, NUM_VEC)) vec8 views; TE-wide view for the K tail
            # (TE = 2 at LPD=32; wider tail slices for sub-warp dots).
            TE: cutlass.Constexpr = (K - WARP * VEC) // LPD
            gA_vec = cute.logical_divide(mA, (None, VEC))
            gW_vec = cute.logical_divide(mW, (None, VEC))
            # Contiguous 32B main slices for the 256-bit (W256) variant.
            EMAIN: cutlass.Constexpr = WARP * VEC // LPD
            gA_v16 = cute.logical_divide(mA, (None, EMAIN))
            gW_v16 = cute.logical_divide(mW, (None, EMAIN))
            gW_tail = cute.logical_divide(
                cute.domain_offset((0, WARP * VEC), mW), (None, TE)
            )
            gA_tail = cute.logical_divide(
                cute.domain_offset((0, WARP * VEC), mA), (None, TE)
            )

            smem = cutlass.utils.SmemAllocator()
            ROWS: cutlass.Constexpr = CW * HC * MB
            # Padded stride LPD+1 keeps the scalar column scatter (STS.32) and
            # the row gather (LDS.32) bank-conflict-free (33 at LPD=32, 9 at
            # LPD=8). A stride-36 variant that autovectorizes the row gather to
            # LDS.128 measured ~1us SLOWER at M=32 (revert if retried).
            COLS: cutlass.Constexpr = LPD + 1
            sm_stage = smem.allocate_tensor(
                cutlass.Float32,
                cute.make_layout(
                    (2, CH, ROWS, COLS),
                    stride=(CH * ROWS * COLS, ROWS * COLS, COLS, 1),
                ),
                byte_alignment=16,
            )

            # Preload this warp's CW*HC weight rows into registers. Weights are
            # long-lived, so this is issued before the PDL wait to overlap the
            # cold-DRAM stream with the predecessor kernel. All LDGs are issued
            # back-to-back, and their conversion is deferred (converting eagerly
            # stalls on the weight data). Fragments must be standalone (VEC,)
            # rmem tensors -- copying into a *slice* of a larger rmem tensor
            # scalarizes the gmem loads (measured on v1). Weights are read once
            # per CTA. .cg loads (LoadCacheMode.GLOBAL, L2-only) were tried to
            # keep weights out of L1: neutral at M<=4, worse at M=8/16 --
            # reverted.
            if const_expr(LPD == WARP):
                wr = cute.make_rmem_tensor((CW, HC, VEC + 2), cutlass.Float32)
                wr_bf16 = []
                for c in cutlass.range_constexpr(CW):
                    for s in cutlass.range_constexpr(HC):
                        row = s * HC_DIM + gw * CW + c
                        w0 = cute.make_rmem_tensor_like(gW_vec[row, (None, lane)])
                        cute.autovec_copy(gW_vec[row, (None, lane)], w0)
                        wr_bf16.append(w0)
                        w1 = cute.make_rmem_tensor_like(gW_tail[row, (None, lane)])
                        cute.autovec_copy(gW_tail[row, (None, lane)], w1)
                        wr_bf16.append(w1)
            else:
                # Sub-warp dot: group g owns PPG (channel, stream) pairs
                # {PPG*g, ...}; lane j covers vec8 k-groups {j + i*LPD} plus
                # a TE-wide tail slice (interleaved split). Same 80 fp32
                # weight regs/lane as LPD=32 for the shipped cfg.
                PPG: cutlass.Constexpr = CW * HC * LPD // WARP
                NV8: cutlass.Constexpr = WARP // LPD
                g = lane // LPD
                j = lane % LPD
                if const_expr(FMA2):
                    # ((stream-pair), (2, ELEMS)) nested layout with the
                    # stream dim stride-1, so (None, v) slice loads emit
                    # vector<2> -- the structural f32x2 operand pairing.
                    ELE: cutlass.Constexpr = NV8 * VEC + TE
                    wr = cute.make_rmem_tensor(
                        cute.make_layout(
                            (PPG // 2, (2, ELE)),
                            stride=(2 * ELE, (1, 2)),
                        ),
                        cutlass.Float32,
                    )
                else:
                    wr = cute.make_rmem_tensor((PPG, NV8 * VEC + TE), cutlass.Float32)
                wr_bf16 = []
                for r in cutlass.range_constexpr(PPG):
                    pr = PPG * g + r
                    row = (pr % HC) * HC_DIM + gw * CW + pr // HC
                    if const_expr(W256):
                        # One 32B fragment per row (contiguous slice).
                        frag = cute.make_rmem_tensor_like(gW_v16[row, (None, j)])
                        self._wload(gW_v16[row, (None, j)], frag, WNA)
                        wr_bf16.append(frag)
                    else:
                        for i in cutlass.range_constexpr(NV8):
                            frag = cute.make_rmem_tensor_like(
                                gW_vec[row, (None, j + i * LPD)]
                            )
                            self._wload(gW_vec[row, (None, j + i * LPD)], frag, WNA)
                            wr_bf16.append(frag)
                    tf = cute.make_rmem_tensor_like(gW_tail[row, (None, j)])
                    self._wload(gW_tail[row, (None, j)], tf, WNA)
                    wr_bf16.append(tf)

            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            # Lane l owns (channel c_l, stream s_l, token offset ml_l) in the
            # epilogue; its xn value is loaded one block ahead of use.
            c_l = lane // (HC * MB)
            s_l = (lane % (HC * MB)) // MB
            ml_l = lane % MB
            xn_col = s_l * HC_DIM + gw * CW + c_l
            xn_prev = cutlass.Float32(0.0)
            m_x0 = msub * MB + ml_l
            if m_x0 < M:
                xn_prev = mXn[m_x0, xn_col].to(cutlass.Float32)

            # Widen the weights (the a-slices are read straight from gmem in the
            # dot loop; the 20KB working set sits in L1/L2 across all CTAs).
            if const_expr(LPD == WARP):
                for c in cutlass.range_constexpr(CW):
                    for s in cutlass.range_constexpr(HC):
                        w0_f32 = wr_bf16[2 * (c * HC + s)].load().to(cutlass.Float32)
                        w1_f32 = (
                            wr_bf16[2 * (c * HC + s) + 1].load().to(cutlass.Float32)
                        )
                        for v in cutlass.range_constexpr(VEC):
                            wr[c, s, v] = w0_f32[v]
                        for v in cutlass.range_constexpr(2):
                            wr[c, s, VEC + v] = w1_f32[v]
            elif const_expr(W256):
                for r in cutlass.range_constexpr(PPG):
                    w_f32 = wr_bf16[2 * r].load().to(cutlass.Float32)
                    for v in cutlass.range_constexpr(EMAIN):
                        wr[r, v] = w_f32[v]
                    t_f32 = wr_bf16[2 * r + 1].load().to(cutlass.Float32)
                    for v in cutlass.range_constexpr(TE):
                        wr[r, EMAIN + v] = t_f32[v]
            else:
                for r in cutlass.range_constexpr(PPG):
                    for i in cutlass.range_constexpr(NV8):
                        w_f32 = wr_bf16[r * (NV8 + 1) + i].load().to(cutlass.Float32)
                        for v in cutlass.range_constexpr(VEC):
                            if const_expr(FMA2):
                                wr[r // 2, (r % 2, i * VEC + v)] = w_f32[v]
                            else:
                                wr[r, i * VEC + v] = w_f32[v]
                    t_f32 = wr_bf16[r * (NV8 + 1) + NV8].load().to(cutlass.Float32)
                    for v in cutlass.range_constexpr(TE):
                        if const_expr(FMA2):
                            wr[r // 2, (r % 2, NV8 * VEC + v)] = t_f32[v]
                        else:
                            wr[r, NV8 * VEC + v] = t_f32[v]

            if const_expr(LPD == WARP):
                acc = cute.make_rmem_tensor((CW, HC, MB), cutlass.Float32)
            else:
                acc = cute.make_rmem_tensor((PPG, MB), cutlass.Float32)

            # This CTA processes M-blocks msub, msub+M_SPLITS, msub+2*M_SPLITS,
            # ... (M-split grid doubles warp supply at M>=2*MB to hide gmem
            # latency; CTAs past M have already exited above).
            # par = per-CTA block index & 1 (staging double buffer).
            first_m0 = msub * MB
            nblk = cute.ceil_div(M, MB)
            rem = nblk - 1 - msub
            b_last = msub + (max(rem, cutlass.Int32(0)) // M_SPLITS) * M_SPLITS

            # Prologue: first block dots + scatter.
            self._dot_block(
                acc,
                gA_vec,
                gA_v16,
                gA_tail,
                wr,
                first_m0,
                M,
                lane,
                CW,
                HC,
                MB,
                VEC,
                K,
                LPD,
                FMA2,
                W256,
            )
            self._scatter(
                acc,
                sm_stage,
                mOut,
                first_m0,
                cutlass.Int32(0),
                gw,
                M,
                wid,
                lane,
                CW,
                HC,
                MB,
                LPD,
                DEBUG_MODE,
            )

            # Steady state: dots of block k, then reduce+epilogue of block k-1.
            for m0 in cutlass.range(
                first_m0 + M_SPLITS * MB, M, M_SPLITS * MB, unroll=2
            ):
                par = ((m0 // MB - msub) // M_SPLITS) & 1
                xn_cur = cutlass.Float32(0.0)
                m_x = m0 + ml_l
                if m_x < M:
                    xn_cur = mXn[m_x, xn_col].to(cutlass.Float32)
                self._dot_block(
                    acc,
                    gA_vec,
                    gA_v16,
                    gA_tail,
                    wr,
                    m0,
                    M,
                    lane,
                    CW,
                    HC,
                    MB,
                    VEC,
                    K,
                    LPD,
                    FMA2,
                    W256,
                )
                self._scatter(
                    acc,
                    sm_stage,
                    mOut,
                    m0,
                    par,
                    gw,
                    M,
                    wid,
                    lane,
                    CW,
                    HC,
                    MB,
                    LPD,
                    DEBUG_MODE,
                )
                if const_expr(DEBUG_MODE != 2):
                    prev_m0 = m0 - M_SPLITS * MB
                    self._reduce_epilogue(
                        sm_stage,
                        mOut,
                        prev_m0,
                        xn_prev,
                        1 - par,
                        gw,
                        M,
                        wid,
                        lane,
                        CW,
                        HC,
                        MB,
                        WARP,
                        LPD,
                        DEBUG_MODE,
                    )
                xn_prev = xn_cur

            # Final block's reduce + epilogue.
            if const_expr(DEBUG_MODE != 2):
                m_last = b_last * MB
                par_last = ((b_last - msub) // M_SPLITS) & 1
                self._reduce_epilogue(
                    sm_stage,
                    mOut,
                    m_last,
                    xn_prev,
                    par_last,
                    gw,
                    M,
                    wid,
                    lane,
                    CW,
                    HC,
                    MB,
                    WARP,
                    LPD,
                    DEBUG_MODE,
                )
