# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original workbench kernel for the Qwen4Exp mHC UP projection, fusing the
semantics of vllm/models/qwen4_exp/nvidia/ops/hc.py::_hc_gate_mix_kernel into
the up GEMM itself.

out[m, h] = (1/HC) * sum_s sigmoid(bf16(gate_s[m])) * xn[m, s*HC_DIM + h]
    gate_s[m] = dot(a[m, :], w[s*HC_DIM + h, :])   (fp32 accumulation)

The bf16 rounding of each gate (the production GEMM output boundary) is
preserved: fp32 dot -> bf16 -> fp32 -> sigmoid. The [M, N] gate tensor is
never materialized.

Decomposition (GB300-oriented): one CTA per group of CH output channels
(HC_DIM/CH CTAs), one warp per channel. Each warp preloads its channel's HC
weight rows (HC x 40 x 16B over 32 lanes) into registers, then loops over
tokens: load the token's a row once (16B per lane x 2 rounds), 4 warp-local
dots + shuffle reductions, gates staged via SMEM; xn is prefetched before the
token loop since it does not depend on the gates. One thread per (token,
channel) applies the epilogue. M is dynamic (<= MAX_M).
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr


def _sigmoid_f32(x):
    return 1.0 / (1.0 + cute.math.exp(x * (-1.0)))


class UpGateMixKernel:
    """Fused up-projection + sigmoid gate mix.

    :compile-key: shape-static except M; a single compilation covers all
        M <= max_m. ``channels_per_cta`` trades grid size against per-CTA
        work and is a compile-time constant.
    """

    def __init__(
        self,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        max_m: int = 256,
        vec_width: int = 8,
        channels_per_cta: int = 4,
        use_pdl: bool = False,
    ):
        self.k = k
        self.n = n
        self.hc = hc
        self.hc_dim = n // hc
        self.max_m = max_m
        self.vec_width = vec_width
        self.ch = channels_per_cta
        self.use_pdl = use_pdl
        self.num_warps = channels_per_cta  # one warp per channel
        self.num_threads = self.num_warps * cute.arch.WARP_SIZE
        assert n % hc == 0
        assert self.hc_dim % channels_per_cta == 0

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
            self.vec_width,
            self.ch,
            self.num_threads,
        ).launch(
            grid=[self.hc_dim // self.ch, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self.ch * self.hc * self.max_m * 4,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

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
        VEC: cutlass.Constexpr,
        CH: cutlass.Constexpr,
        NUM_THREADS: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        h0, _, _ = cute.arch.block_idx()
        wid = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        WARP: cutlass.Constexpr = cute.arch.WARP_SIZE
        NUM_VEC: cutlass.Constexpr = K // VEC  # 40
        RAGGED: cutlass.Constexpr = NUM_VEC - WARP  # 8 lanes load a 2nd vec

        M = cute.size(mA, mode=[0])
        h = h0 * CH + wid  # this warp's output channel

        # (M, (VEC, NUM_VEC)) main view: one vec8 per lane, all 32 lanes.
        gA_vec = cute.logical_divide(mA, (None, VEC))
        gW_vec = cute.logical_divide(mW, (None, VEC))
        # Constexpr-offset tail view (K = 256 + 64): plain `lane` indexing
        # keeps the loads vectorized (dynamic `lane + WARP` scalarizes).
        gA_tail = cute.logical_divide(
            cute.domain_offset((0, WARP * VEC), mA), (None, VEC)
        )
        gW_tail = cute.logical_divide(
            cute.domain_offset((0, WARP * VEC), mW), (None, VEC)
        )

        # Per-(channel, stream, token) gate scratch.
        smem = cutlass.utils.SmemAllocator()
        sm_gate = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout((CH, HC, MAX_M), stride=(HC * MAX_M, MAX_M, 1)),
            byte_alignment=16,
        )

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # Preload this channel's HC weight rows into registers. Fragments must
        # be standalone (VEC,) rmem tensors -- copying into a *slice* of a
        # larger rmem tensor scalarizes the gmem loads.
        # wr_f32[s][round]: fp32 vectors; round 1 is the ragged tail.
        wr_f32 = []
        for s in cutlass.range_constexpr(HC):
            row = s * HC_DIM + h
            w0 = cute.make_rmem_tensor_like(gW_vec[row, (None, lane)])
            cute.autovec_copy(gW_vec[row, (None, lane)], w0)
            w1 = cute.make_rmem_tensor_like(gW_vec[row, (None, lane)])
            w1.fill(0.0)
            if lane < RAGGED:
                cute.autovec_copy(gW_tail[row, (None, lane)], w1)
            wr_f32.append(
                (w0.load().to(cutlass.Float32), w1.load().to(cutlass.Float32))
            )

        # xn does not depend on the gates: prefetch this thread's epilogue
        # values (channel ch = tidx % CH, token = tidx // CH) before the
        # token loop so the tail has no serial gmem latency.
        TOK0: cutlass.Constexpr = NUM_THREADS // CH  # tokens per pass
        xn_frag = cute.make_rmem_tensor((HC,), cutlass.Float32)
        xn_frag.fill(0.0)
        e_ch = tidx % CH
        e_m = tidx // CH
        e_h = h0 * CH + e_ch
        if e_m < M:
            for s in cutlass.range_constexpr(HC):
                xn_frag[s] = mXn[e_m, s * HC_DIM + e_h].to(cutlass.Float32)

        for m in cutlass.range(M, unroll=1):
            av0 = gA_vec[m, (None, lane)]
            ar0 = cute.make_rmem_tensor_like(av0)
            cute.autovec_copy(av0, ar0)
            ar0_f32 = ar0.load().to(cutlass.Float32)
            ar1 = cute.make_rmem_tensor_like(av0)
            ar1.fill(0.0)
            if lane < RAGGED:
                cute.autovec_copy(gA_tail[m, (None, lane)], ar1)
            ar1_f32 = ar1.load().to(cutlass.Float32)

            for s in cutlass.range_constexpr(HC):
                acc = cutlass.Float32(0.0)
                w0_f32, w1_f32 = wr_f32[s]
                for v in cutlass.range_constexpr(VEC):
                    acc += ar0_f32[v] * w0_f32[v] + ar1_f32[v] * w1_f32[v]
                acc = cute.arch.warp_reduction_sum(acc)
                if lane == 0:
                    sm_gate[wid, s, m] = acc

        cute.arch.sync_threads()

        # Epilogue: thread t handles (m = t // CH + i * TOK0, ch = t % CH).
        if e_m < M:
            acc = cutlass.Float32(0.0)
            for s in cutlass.range_constexpr(HC):
                g = sm_gate[e_ch, s, e_m].to(cutlass.BFloat16).to(cutlass.Float32)
                acc += _sigmoid_f32(g) * xn_frag[s]
            mOut[e_m, e_h] = (acc * (1.0 / HC)).to(cutlass.BFloat16)
        for m in cutlass.range(e_m + TOK0, M, TOK0, unroll=1):
            acc = cutlass.Float32(0.0)
            for s in cutlass.range_constexpr(HC):
                g = sm_gate[e_ch, s, m].to(cutlass.BFloat16).to(cutlass.Float32)
                x = mXn[m, s * HC_DIM + e_h].to(cutlass.Float32)
                acc += _sigmoid_f32(g) * x
            mOut[m, e_h] = (acc * (1.0 / HC)).to(cutlass.BFloat16)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
