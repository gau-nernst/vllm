# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fourth fused mHC UP kernel variant: mma.sync m16n8k16 with swapped operands.

Same math as `_up_gate_mix.py`:

out[m, h] = (1/HC) * sum_s sigmoid(bf16(gate_s[m])) * xn[m, s*HC_DIM + h]
    gate_s[m] = dot(a[m, :], w[s*HC_DIM + h, :])   (fp32 accumulation)

Decomposition (GB300): the MMA operands are swapped relative to a plain GEMM --
output channels sit on the MMA-M (16) dim and tokens on the MMA-N (8) dim, so
each mma.sync.m16n8k16 computes a gate^T tile C[16 channels, 8 tokens] =
W_tile[16, k] @ A_tile[8 tokens, k]^T. At M(tokens) <= 8 only the N=8 tile is
padded (activations are tiny); channels pack densely along the 16-dim so no
weight bandwidth is wasted.

Geometry: one CTA owns 16 output channels and has HC=4 warps -- warp s computes
stream s's gate tile for those channels (one warp per stream; a 1-warp-per-CTA
prototype measured 1.0 active warps/scheduler and could not hide cp.async /
ldmatrix latency). Grid = HC_DIM/16 = 160 CTAs = 640 warps. M > 8 uses NT =
max_m/8 token tiles, all of them as ONE MMA-N dimension: a single B fragment
per k16 block covers every token tile (one ldmatrix copy + one gemm per
k-block instead of NT serialized copy->MMA pairs). An optional ksplit factor
assigns each warp a K/ksplit slice (fp32 partials summed in the exchange);
measured neutral-to-slower, kept for experimentation.

Per-warp weight pipeline: cp.async (.cg, bypasses L1) streams the warp's
16ch x K weight rows through num_stages SMEM stages; the first num_stages-1
stages are issued BEFORE the PDL wait to overlap the cold-DRAM stream with the
predecessor kernel. cp.async commit groups are per-thread; stage kt is
consumed under wait_group(min(ST-1, NUM_KT-1-kt)) -- pending groups newer than
stage kt's group, which is exact at both ends of the k-loop (a constant ST-2
wait is over-strict mid-loop and racy on the last stages).

Activations (<= max_m x 320 bf16) and xn (the CTA's 16-channel columns of all
HC streams, M x 64 bf16) are cooperatively cp.async-staged into SMEM once per
CTA after the PDL wait; the epilogue reads xn from SMEM (per-lane 2B gmem
gathers measured +5us at M=1). After the mainloop, each warp writes its fp32
gate accumulators to a padded SMEM exchange buffer; the epilogue then runs one
thread per (channel, token): bf16-round gate -> sigmoid -> x xn ->
stream-serial sum (production order) -> x 1/HC -> bf16 store.

M is dynamic (<= max_m); grid is static, so the kernel is CUDA-graph safe.
"""

import math

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr
from cutlass.cute.nvgpu import cpasync, warp


def _sigmoid_f32(x):
    # fastmath: bf16 gate rounding (rel ~4e-3) dominates the ~1e-7 fast
    # exp/rcp error, and the output is bf16 anyway.
    return cute.math.rcp(1.0 + cute.math.exp(x * (-1.0), fastmath=True), fastmath=True)


class UpGateMixV4Kernel:
    """Fused up-projection + sigmoid gate mix (mma.sync, swapped operands).

    :compile-key: shape-static except M; a single compilation covers all
        M <= max_m.
    """

    def __init__(
        self,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        max_m: int = 32,
        tile_k: int = 64,
        num_stages: int = 5,
        ksplit: int = 1,
        use_pdl: bool = False,
        debug_mode: int = 0,  # 0=full, 1=gate only, 2=loads+mma, 3=loads only
        min_blocks_per_mp: int = 1,
    ):
        self.k = k
        self.n = n
        self.hc = hc
        self.hc_dim = n // hc
        self.max_m = max_m
        self.nt = max_m // 8  # token tiles of 8 (MMA-N)
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.ksplit = ksplit  # K-slices per stream (shortens per-warp chains)
        self.kpb = tile_k // 16  # k16 blocks per pipeline stage
        self.num_kt = k // tile_k
        self.num_kt_w = self.num_kt // ksplit  # stages per warp
        self.use_pdl = use_pdl
        self.debug_mode = debug_mode
        self.min_blocks_per_mp = min_blocks_per_mp
        self.num_warps = hc * ksplit  # warp = (k-slice, stream)
        self.num_threads = self.num_warps * cute.arch.WARP_SIZE
        assert n % hc == 0 and self.hc_dim % 16 == 0
        assert k % tile_k == 0 and tile_k % 16 == 0
        assert self.num_kt % ksplit == 0
        assert self.num_kt_w >= num_stages >= 2
        assert max_m % 16 == 0  # activation/xn tiled copies cover 16 rows
        # 16B-aligned cp.async rows: K and tile_k multiples of 8 bf16.
        assert k % 8 == 0

    def _make_smem_layout(self, dtype, smem_tiler):
        """Staged swizzled SMEM layout.

        NOTE: CuTe DSL make_swizzle is BYTE-domain (unlike element-domain
        C++ CuTe): a 128B-row swizzle for 16B copy atoms is S<3,4,3>
        (quack/copy_utils.py documents the byte<->element recast).
        """
        row_bytes = min(smem_tiler[1], 64) * dtype.width // 8
        # 16B copy/ldmatrix atoms: XOR the row index into the 16B-quad bits
        # (byte-domain Swizzle<B,4,3> with B = log2(row_bytes/16)) so the 8
        # rows of an ldmatrix matrix land on distinct bank quads.
        swizzle_bits = max(0, min(int(math.log2(row_bytes)) - 4, 3))
        major = min(smem_tiler[1], 64)
        layout_atom_outer = cute.make_layout((8, major), stride=(major, 1))
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 4, 3), 0, layout_atom_outer
        )
        return cute.tile_to_shape(
            layout_atom, smem_tiler, tuple(range(len(smem_tiler)))
        )

    def _smem_bytes(self) -> int:
        def round128(x: int) -> int:
            return (x + 127) // 128 * 128

        sa = 8 * self.nt * self.k * 2
        sw = 16 * self.tile_k * self.num_stages * 2
        sxn = 8 * self.nt * self.hc * 16 * 2
        sgates = self.ksplit * self.hc * self.nt * 16 * 9 * 4
        return (
            round128(sa)
            + self.hc * self.ksplit * round128(sw)
            + round128(sxn)
            + round128(sgates)
        )

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,  # [M, K] bf16
        mW: cute.Tensor,  # [N, K] bf16
        mXn: cute.Tensor,  # [M, N] bf16
        mOut: cute.Tensor,  # [M, HC_DIM] bf16
        stream: CUstream,
    ):
        NT: cutlass.Constexpr = self.nt
        sA_layout = self._make_smem_layout(mA.element_type, (8 * NT, self.k))
        sW_layout = self._make_smem_layout(
            mW.element_type, (16, self.tile_k, self.num_stages)
        )

        atom_g2s = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mA.element_type,
            num_bits_per_copy=128,
        )  # cp.async GMEM -> SMEM, bypassing L1
        atom_g2s_w = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            mW.element_type,
            num_bits_per_copy=128,
        )  # weights are streamed once; evict-first L2 policy
        # Weights: (16, tile_k) tiles per warp; lanes laid across K.
        k_threads = self.tile_k // 8
        tiled_copy_W = cute.make_tiled_copy_tv(
            atom_g2s_w,
            cute.make_layout(
                (cute.arch.WARP_SIZE // k_threads, k_threads), stride=(k_threads, 1)
            ),
            cute.make_layout((1, 8)),
        )
        # Activations: (8*NT, K) tile staged by all 128 threads (16 rows x 64
        # K-elems per pass).
        tiled_copy_A = cute.make_tiled_copy_tv(
            atom_g2s,
            cute.make_layout((16, 8), stride=(8, 1)),
            cute.make_layout((1, 8)),
        )
        # xn: (8*NT tokens, HC streams, 16 channels) gather of 32B segments.
        tiled_copy_X = cute.make_tiled_copy_tv(
            atom_g2s,
            cute.make_layout((16, 4, 2), stride=(8, 2, 1)),
            cute.make_layout((1, 1, 8)),
        )
        op = warp.MmaF16BF16Op(mA.element_type, cutlass.Float32, (16, 8, 16))
        tiled_mma = cute.make_tiled_mma(
            op, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 8, 16)
        )

        self.kernel(
            mA,
            mW,
            mXn,
            mOut,
            sA_layout,
            sW_layout,
            tiled_copy_A,
            tiled_copy_W,
            tiled_copy_X,
            tiled_mma,
        ).launch(
            grid=[self.hc_dim // 16, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_bytes(),
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mW: cute.Tensor,
        mXn: cute.Tensor,
        mOut: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sW_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_W: cute.TiledCopy,
        tiled_copy_X: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warpid = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        K: cutlass.Constexpr = self.k
        HC: cutlass.Constexpr = self.hc
        HC_DIM: cutlass.Constexpr = self.hc_dim
        NT: cutlass.Constexpr = self.nt
        KT: cutlass.Constexpr = self.tile_k
        ST: cutlass.Constexpr = self.num_stages
        KS: cutlass.Constexpr = self.ksplit
        KPB: cutlass.Constexpr = self.kpb
        NUM_KT: cutlass.Constexpr = self.num_kt_w  # k-stages per warp
        DEBUG_MODE: cutlass.Constexpr = self.debug_mode

        M = cute.size(mA, mode=[0])
        ch0 = bidx * 16  # this CTA owns output channels [ch0, ch0+16)
        # warp = (k-slice ks, stream s): each warp computes a K/KS partial
        # gate tile; the fp32 partials are summed in the SMEM exchange.
        ks = warpid // HC
        s = warpid % HC

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(
            mA.element_type,
            sA_layout.outer,
            byte_alignment=128,
            swizzle=sA_layout.inner,
        )
        sW_cosize: cutlass.Constexpr = 16 * self.tile_k * self.num_stages
        sW_flat = smem.allocate_tensor(
            mW.element_type,
            cute.make_layout(HC * KS * sW_cosize),
            byte_alignment=128,
        )
        # this warp's stream tile (composed swizzled layout, offset by stream);
        # the dynamic offset drops inferred alignment, so restate it (the
        # per-warp footprint is a multiple of 128B by construction).
        sW_ptr = cute.make_ptr(
            mW.element_type,
            (sW_flat.iterator + warpid * sW_cosize).toint(),
            cute.AddressSpace.smem,
            assumed_align=128,
        )
        sW_w = cute.make_tensor(sW_ptr, sW_layout)
        sXn = smem.allocate_tensor(
            mXn.element_type,
            cute.make_layout((8 * NT, HC, 16), stride=(HC * 16, 16, 1)),
            byte_alignment=128,
        )
        # +1 padding on the token dim keeps the epilogue reads conflict-free.
        sGates = smem.allocate_tensor(
            cutlass.Float32,
            cute.make_layout(
                (KS, HC, NT, 16, 9), stride=(HC * NT * 144, NT * 144, 144, 9, 1)
            ),
            byte_alignment=128,
        )

        # --- per-warp weight pipeline (warp = (k-slice, stream)) ---
        thr_W = tiled_copy_W.get_slice(lane)
        tWgW = [
            thr_W.partition_S(
                cute.local_tile(
                    mW, (16, KT), (s * (HC_DIM // 16) + bidx, ks * NUM_KT + kt)
                )
            )
            for kt in range(NUM_KT)
        ]
        tWsW = thr_W.partition_D(sW_w)

        # --- cooperative activation + xn staging (128 threads) ---
        thr_A = tiled_copy_A.get_slice(tidx)
        gA = cute.local_tile(mA, (8 * NT, K), (0, 0))
        tAgA = thr_A.partition_S(gA)
        tAsA = thr_A.partition_D(sA)
        tAcA = thr_A.partition_S(cute.make_identity_tensor((8 * NT, K)))
        NUM_VEC: cutlass.Constexpr = cute.size(tAgA, mode=[0, 1])
        RM: cutlass.Constexpr = cute.size(tAgA, mode=[1])
        RK: cutlass.Constexpr = cute.size(tAgA, mode=[2])

        thr_X = tiled_copy_X.get_slice(tidx)
        gXn_ptr = cute.make_ptr(
            mXn.element_type,
            (mXn.iterator + ch0).toint(),
            cute.AddressSpace.gmem,
            assumed_align=32,
        )
        gXn = cute.make_tensor(
            gXn_ptr,
            cute.make_layout((8 * NT, HC, 16), stride=(self.n, HC_DIM, 1)),
        )
        tXgX = thr_X.partition_S(gXn)
        tXsX = thr_X.partition_D(sXn)
        tXcX = thr_X.partition_S(cute.make_identity_tensor((8 * NT, HC, 16)))
        NXV: cutlass.Constexpr = cute.size(tXgX, mode=[0, 1])
        XRM: cutlass.Constexpr = cute.size(tXgX, mode=[1])

        # MMA partitions. All NT token tiles are one MMA-N dimension: a
        # single B fragment per k16 block covers every token tile, which
        # quarters the ldmatrix copy count and the serialized copy->MMA
        # chains in the mainloop.
        thr_mma = tiled_mma.get_slice(lane)
        tCsW = thr_mma.partition_A(sW_w)
        tCsB = thr_mma.partition_B(sA)
        frag_c_shape = tiled_mma.partition_shape_C((16, 8 * NT))
        acc = tiled_mma.make_fragment_C(frag_c_shape)
        acc.fill(0.0)

        # ldmatrix SMEM -> register copy paths.
        atom_s2r = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(False, 4), mA.element_type
        )
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(lane)
        thr_s2r_B = tiled_s2r_B.get_slice(lane)
        tCsW_v = thr_s2r_A.partition_S(sW_w)
        tCrW = tiled_mma.make_fragment_A(tCsW[None, None, None, 0])
        tCrW_v = thr_s2r_A.retile(tCrW)
        tCsB_v = thr_s2r_B.partition_S(sA)
        frag_b_shape = tiled_mma.partition_shape_B((8 * NT, 16))  # one k16
        tCrB = [tiled_mma.make_fragment_B(frag_b_shape) for _ in range(KPB)]
        tCrB_v = [thr_s2r_B.retile(f) for f in tCrB]

        # --- prologue: prefetch weight stages 0..ST-2 (w is a persistent
        # weight, safe to read before the PDL wait) ---
        for st in cutlass.range_constexpr(ST - 1):
            cute.copy(tiled_copy_W, tWgW[st], tWsW[None, None, None, st])
            cute.arch.cp_async_commit_group()

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        # --- stage activations + xn (M-predicated rows; first 128 threads) ---
        if tidx < 128:
            tApA = cute.make_rmem_tensor(
                cute.make_layout((NUM_VEC, RM, RK)), cutlass.Boolean
            )
            for v in cutlass.range_constexpr(NUM_VEC):
                for rm in cutlass.range_constexpr(RM):
                    for rk in cutlass.range_constexpr(RK):
                        tApA[v, rm, rk] = cute.elem_less(tAcA[((0, v), rm, rk)], (M, K))
            cute.copy(tiled_copy_A, tAgA, tAsA, pred=tApA)

            tXpX = cute.make_rmem_tensor(
                cute.make_layout((NXV, XRM, 1, 1)), cutlass.Boolean
            )
            for v in cutlass.range_constexpr(NXV):
                for rm in cutlass.range_constexpr(XRM):
                    tXpX[v, rm, 0, 0] = cute.elem_less(
                        tXcX[((0, v), rm, 0, 0)], (M, HC, 16)
                    )
            cute.copy(tiled_copy_X, tXgX, tXsX, pred=tXpX)
        cute.arch.cp_async_commit_group()

        # kt=0: issue stage ST-1, then drain everything older (incl. the
        # activations) while leaving that newest weight stage in flight.
        if const_expr(ST - 1 < NUM_KT):
            cute.copy(tiled_copy_W, tWgW[ST - 1], tWsW[None, None, None, (ST - 1) % ST])
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(1)
        cute.arch.sync_threads()  # cross-warp visibility of sA / sXn

        # --- mainloop over K stages (fully unrolled) ---
        for kt in cutlass.range_constexpr(NUM_KT):
            if kt > 0:
                kt_next = kt + ST - 1
                if const_expr(kt_next < NUM_KT):
                    cute.copy(
                        tiled_copy_W,
                        tWgW[kt_next],
                        tWsW[None, None, None, kt_next % ST],
                    )
                    cute.arch.cp_async_commit_group()
                # exact in-flight allowance: groups newer than stage kt's
                wait: cutlass.Constexpr = min(ST - 1, NUM_KT - 1 - kt)
                cute.arch.cp_async_wait_group(wait)
                cute.arch.sync_warp()
            st = kt % ST
            if const_expr(DEBUG_MODE != 3):
                cute.copy(tiled_s2r_A, tCsW_v[None, None, None, st], tCrW_v)
                for kb in cutlass.range_constexpr(KPB):
                    cute.copy(
                        tiled_s2r_B,
                        tCsB_v[None, None, ks * NUM_KT * KPB + kt * KPB + kb],
                        tCrB_v[kb][None, None, 0],
                    )
                    cute.gemm(
                        tiled_mma,
                        acc,
                        tCrW[None, None, kb],
                        tCrB[kb][None, None, 0],
                        acc,
                    )

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # --- cross-warp gate exchange ---
        # acc element i: row = lane//4 + 8*(i//2) (channel), col =
        # (lane%4)*2 + i%2 (token). Each (ks, s) warp deposits its fp32
        # K/KS partial.
        for nt in cutlass.range_constexpr(NT):
            for i in cutlass.range_constexpr(4):
                row = lane // 4 + 8 * (i // 2)
                col = (lane % 4) * 2 + (i % 2)
                sGates[ks, s, nt, row, col] = acc[i, 0, nt]
        cute.arch.sync_threads()

        # --- epilogue: one thread per (channel, token); 128 active ---
        ch = tidx % 16
        tok = tidx // 16
        if const_expr(DEBUG_MODE >= 2):
            if warpid == 0 and lane < 16:
                mOut[0, ch0 + lane] = sGates[0, 0, 0, lane, 0].to(cutlass.BFloat16)
        elif tidx < 8 * 16:
            for nt in cutlass.range_constexpr(NT):
                m = nt * 8 + tok
                if m < M:
                    if const_expr(DEBUG_MODE == 1):
                        mOut[m, ch0 + ch] = sGates[0, 0, nt, ch, tok].to(
                            cutlass.BFloat16
                        )
                    else:
                        # Batch all SMEM loads before the math so their
                        # latencies overlap, then the production rounding
                        # boundary: fp32 gate -> bf16 -> fp32 -> sigmoid,
                        # stream-serial mix.
                        gates = []
                        xns = []
                        for s_ in cutlass.range_constexpr(HC):
                            gate = cutlass.Float32(0.0)
                            for k_ in cutlass.range_constexpr(KS):
                                gate += sGates[k_, s_, nt, ch, tok]
                            gates.append(gate)
                            xns.append(sXn[m, s_, ch].to(cutlass.Float32))
                        out_acc = cutlass.Float32(0.0)
                        for s_ in cutlass.range_constexpr(HC):
                            g = gates[s_].to(cutlass.BFloat16).to(cutlass.Float32)
                            out_acc += _sigmoid_f32(g) * xns[s_]
                        mOut[m, ch0 + ch] = (out_acc * (1.0 / HC)).to(cutlass.BFloat16)
