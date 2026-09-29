# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Second fused mHC UP kernel variant: lane = (token, k-slice).

Same math as `_up_gate_mix.py`:

out[m, h] = (1/HC) * sum_s sigmoid(bf16(gate_s[m])) * xn[m, s*HC_DIM + h]
    gate_s[m] = dot(a[m, :], w[s*HC_DIM + h, :])   (fp32 accumulation)

Decomposition: one warp per output channel as before, but the 32 lanes are
split as 4 tokens x 8 k-lanes (K=320 = 8 lanes x 40 elems = 8 x 5 vec8, no
ragged tail). Each warp preloads its channel's HC weight k-slices into
registers (4 rows x 40 elems = 160 bf16/lane), then loops over tokens 4 at a
time: each lane loads its token's k-slice (5 x 16B), 4 dots in fp32, and a
3-step butterfly shuffle reduces within the 8-lane k-group. The ks==0 lane
of each token group then applies the sigmoid-mix epilogue directly from
registers -- no SMEM, no block barrier, no full-warp reductions, and the
per-token serial dependency of v1 is gone.

Per-lane register cost is ~160 fp32 for the weights plus ~40 for the token
fragment, which caps occupancy at ~2 CTAs(128T)/SM with CH=4.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr

TOK_PER_WARP = 4
K_LANES = 8


def _sigmoid_f32(x):
    return 1.0 / (1.0 + cute.math.exp(x * (-1.0)))


class UpGateMixV2Kernel:
    """Fused up-projection + sigmoid gate mix (token x k-slice lanes).

    :compile-key: shape-static except M; a single compilation covers all
        M <= max_m.
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
        debug_mode: int = 0,  # 0=full, 1=no epilogue, 2=no dot (loads only)
    ):
        self.k = k
        self.n = n
        self.hc = hc
        self.hc_dim = n // hc
        self.max_m = max_m
        self.vec_width = vec_width
        self.ch = channels_per_cta
        self.use_pdl = use_pdl
        self.debug_mode = debug_mode
        self.num_warps = channels_per_cta  # one warp per channel
        self.num_threads = self.num_warps * cute.arch.WARP_SIZE
        assert n % hc == 0
        assert self.hc_dim % channels_per_cta == 0
        # K=320 = K_LANES lanes x NVEC vec8 chunks.
        assert k % (K_LANES * vec_width) == 0
        self.nvec = k // (K_LANES * vec_width)  # 5

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
            self.nvec,
            self.num_threads,
            self.debug_mode,
        ).launch(
            grid=[self.hc_dim // self.ch, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=0,
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
        NVEC: cutlass.Constexpr,
        NUM_THREADS: cutlass.Constexpr,
        DEBUG_MODE: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        h0, _, _ = cute.arch.block_idx()
        wid = cute.arch.warp_idx()
        lane = cute.arch.lane_idx()
        TOK_SUB = lane // K_LANES  # which of the 4 tokens this lane works on
        KS = lane % K_LANES  # which k-slice (40 elems) this lane owns
        TPB: cutlass.Constexpr = TOK_PER_WARP

        M = cute.size(mA, mode=[0])
        h = h0 * CH + wid  # this warp's output channel

        # (rows, (VEC, NUM_VEC)) view: one vec8 per index.
        gA_vec = cute.logical_divide(mA, (None, VEC))
        gW_vec = cute.logical_divide(mW, (None, VEC))

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # Preload this channel's HC weight k-slices into registers.
        # Standalone (VEC,) fragments: copying into a *slice* of a larger
        # rmem tensor scalarizes the gmem loads (measured).
        # wf[s][v]: fp32 vec for row s, chunk v of this lane's k-slice.
        wf = []
        for s in cutlass.range_constexpr(HC):
            row = s * HC_DIM + h
            frags = []
            for v in cutlass.range_constexpr(NVEC):
                src = gW_vec[row, (None, KS * NVEC + v)]
                f = cute.make_rmem_tensor_like(src)
                cute.autovec_copy(src, f)
                frags.append(f.load().to(cutlass.Float32))
            wf.append(frags)

        for m0 in cutlass.range(0, M, TPB, unroll=2):
            m = m0 + TOK_SUB
            valid = m < M

            # xn does not depend on the gates: issue its loads first so the
            # gmem latency hides under the dot product below.
            xf = [cutlass.Float32(0.0)] * HC
            if KS == 0:
                if valid:
                    for s in cutlass.range_constexpr(HC):
                        xf[s] = mXn[m, s * HC_DIM + h].to(cutlass.Float32)

            af = []
            for v in cutlass.range_constexpr(NVEC):
                src = gA_vec[0, (None, KS * NVEC + v)]
                f = cute.make_rmem_tensor_like(src)
                f.fill(0.0)
                if valid:
                    cute.autovec_copy(gA_vec[m, (None, KS * NVEC + v)], f)
                af.append(f.load().to(cutlass.Float32))

            acc = [cutlass.Float32(0.0)] * HC
            if const_expr(DEBUG_MODE != 2):
                for s in cutlass.range_constexpr(HC):
                    for v in cutlass.range_constexpr(NVEC):
                        a_v = af[v]
                        w_v = wf[s][v]
                        acc_s = acc[s]
                        for e in cutlass.range_constexpr(VEC):
                            acc_s += a_v[e] * w_v[e]
                        acc[s] = acc_s
            else:
                # Loads only: consume the fragments so nothing is DCE'd.
                for v in cutlass.range_constexpr(NVEC):
                    a_v = af[v]
                    acc[0] += a_v[0] + wf[0][v][0]
                    acc[1] += a_v[1] + wf[1][v][1]
                    acc[2] += a_v[2] + wf[2][v][2]
                    acc[3] += a_v[3] + wf[3][v][3]

            # Reduce over the 8-lane k-group (bfly xor 4, 2, 1 stays within
            # the 8-aligned lane group). All 4 reductions are independent.
            for off in cutlass.range_constexpr(3):
                for s in cutlass.range_constexpr(HC):
                    acc[s] += cute.arch.shuffle_sync_bfly(acc[s], offset=1 << (2 - off))

            if KS == 0:
                if valid:
                    if const_expr(DEBUG_MODE == 1):
                        # No epilogue: store the raw gate to defeat DCE.
                        mOut[m, h] = acc[0].to(cutlass.BFloat16)
                    else:
                        out_acc = cutlass.Float32(0.0)
                        for s in cutlass.range_constexpr(HC):
                            g = acc[s].to(cutlass.BFloat16).to(cutlass.Float32)
                            out_acc += _sigmoid_f32(g) * xf[s]
                        mOut[m, h] = (out_acc * (1.0 / HC)).to(cutlass.BFloat16)

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
