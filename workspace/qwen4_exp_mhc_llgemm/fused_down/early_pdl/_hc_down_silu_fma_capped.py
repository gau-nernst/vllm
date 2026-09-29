# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grid-capped variant of _hc_down_silu_fma_early.py for PDL co-scheduling.

The uncapped FMA kernel launches one CTA per output column (324 CTAs at
rank=320, hc=4), which puts 2-3 resident CTAs on every SM. The PDL-dependent
up kernel then cannot be placed until down CTAs drain (~0.9us before down
ends), so the up weight stream barely overlaps down.

This variant gives each CTA ``cols_per_cta`` contiguous columns (grid =
``ceil(N / cols_per_cta)``), statically unrolled so ptxas can overlap the
next column's loads with the current column's FMAs. Per-column math
(reduction order, SiLU epilogue) is unchanged, so output is bit-identical.
Freeing SMs lets the dependent up kernel co-schedule at t=0 and stream its
weights concurrently with down's.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import const_expr


def _sigmoid_f32(x):
    return cute.math.rcp(1.0 + cute.math.exp(-x), approx=True)


class HcDownSiluFmaCapped:
    """Same GEMM + fused SiLU epilogue as HcDownSiluFmaEarly, but with a
    statically unrolled persistent column loop: each CTA owns the
    ``cols_per_cta`` contiguous columns ``bid*J .. bid*J+J-1`` and the grid is
    ``ceil(N/J)`` CTAs.

    The static unroll lets ptxas hoist the next column's loads across the
    current column's FMAs (cross-column latency hiding), which the dynamic
    strided-loop variant cannot do — fewer CTAs must keep more loads in
    flight to sustain HBM bandwidth. Freeing SMs lets the dependent up
    kernel co-schedule at t=0 and stream its weights concurrently with
    down's. Per-column math is unchanged, so output is bit-identical.

    The compile key is ``(M, K, threadblock_size, rank, hc, cols_per_cta)``.
    """

    def __init__(
        self,
        k: int,
        cols_per_cta: int,
        threadblock_size: int = 128,
        main_vec_width: int = 8,
        tail_vec_width: int = 4,
        rank: int = 320,
        hc: int = 4,
        min_blocks_per_mp: int = 1,
    ):
        self.cols_per_cta = cols_per_cta
        self.min_blocks_per_mp = min_blocks_per_mp
        self.threadblock_size = threadblock_size
        self.main_vec_width = main_vec_width
        self.tail_vec_width = tail_vec_width
        self.rank = rank
        self.hc = hc
        self.num_warps = threadblock_size // cute.arch.WARP_SIZE
        self._init_k_tiles(k)

    def _vectorized_elems(self, k_extent: int, vec_width: int) -> int:
        vector_tile = vec_width * self.threadblock_size
        return (k_extent // vector_tile) * vector_tile

    def _init_k_tiles(self, k: int) -> None:
        """Split K into vector loops, scalar rounds, and ragged tail."""
        self.k_main_elems = self._vectorized_elems(k, self.main_vec_width)
        self.k_after_main = k - self.k_main_elems
        self.k_tail_elems = self._vectorized_elems(
            self.k_after_main, self.tail_vec_width
        )
        self.k_done_all = self.k_main_elems + self.k_tail_elems
        self.scalar_rem = k - self.k_done_all
        self.ks_full = self.scalar_rem // self.threadblock_size
        self.ks_part = self.scalar_rem % self.threadblock_size
        self.k_scalar_full = self.ks_full * self.threadblock_size
        self.k_part_offset = self.k_done_all + self.k_scalar_full
        self.main_tiles = self.k_main_elems // (
            self.main_vec_width * self.threadblock_size
        )
        self.tail_tiles = self.k_tail_elems // (
            self.tail_vec_width * self.threadblock_size
        )

    @cute.jit
    def _vector_fma(
        self,
        acc: cute.Tensor,
        tA: cute.Tensor,
        tB: cute.Tensor,
        M: cutlass.Constexpr,
        num_tiles: cutlass.Constexpr,
    ):
        for tile in cutlass.range_constexpr(num_tiles):
            bt = tB[None, tile]
            br = cute.make_rmem_tensor_like(bt)
            cute.autovec_copy(bt, br)
            br_f32 = br.load().to(cutlass.Float32)

            for m in cutlass.range_constexpr(M):
                at = tA[m, None, tile]
                ar = cute.make_rmem_tensor_like(at)
                cute.autovec_copy(at, ar)
                vec_width: cutlass.Constexpr = cute.size(ar)
                for v in cutlass.range_constexpr(vec_width):
                    acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]

    def _make_thread_vector_slice(
        self,
        gA_vec: cute.Tensor,
        gB_vec: cute.Tensor,
        tidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        threadblock_size: cutlass.Constexpr,
    ):
        # (M/N, K_TILE, K_LANE, K_VEC); tidx selects K_LANE.
        tA = cute.logical_divide(gA_vec, (None, (None, threadblock_size)))
        tB = cute.logical_divide(gB_vec, (None, (None, threadblock_size)))
        return tA[None, (None, (tidx, None))], tB[n_idx, (None, (tidx, None))]

    def _make_k_slice(
        self,
        gX: cute.Tensor,
        k_offset: cutlass.Constexpr,
        k_extent: cutlass.Constexpr,
    ):
        k_layout_extent: cutlass.Constexpr = (
            1 if const_expr(k_extent == 0) else k_extent
        )

        if const_expr(k_offset == 0):
            return cute.local_tile(
                gX, (cute.size(gX, mode=[0]), k_layout_extent), (0, 0)
            )
        return cute.local_tile(
            cute.domain_offset((0, k_offset), gX),
            (cute.size(gX, mode=[0]), k_layout_extent),
            (0, 0),
        )

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        M: cutlass.Constexpr,
        K_dim: cutlass.Constexpr,
        N_dim: cutlass.Int32,
        stream: CUstream,
    ):
        self.kernel(
            gA,
            gB,
            gC,
            M,
            N_dim,
            self.main_vec_width,
            self.tail_vec_width,
            self.threadblock_size,
            self.num_warps,
            self.k_main_elems,
            self.k_tail_elems,
            self.k_done_all,
            self.ks_full,
            self.ks_part,
            self.k_scalar_full,
            self.k_part_offset,
            self.main_tiles,
            self.tail_tiles,
            self.rank,
            self.hc,
            self.cols_per_cta,
        ).launch(
            grid=[cute.ceil_div(N_dim, self.cols_per_cta), 1, 1],
            block=[self.threadblock_size, 1, 1],
            smem=M * 4 * self.num_warps,
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=self.min_blocks_per_mp,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gC: cute.Tensor,
        M: cutlass.Constexpr,
        N_dim: cutlass.Int32,
        main_vec_width: cutlass.Constexpr,
        tail_vec_width: cutlass.Constexpr,
        threadblock_size: cutlass.Constexpr,
        num_warps: cutlass.Constexpr,
        k_main_elems: cutlass.Constexpr,
        k_tail_elems: cutlass.Constexpr,
        k_done_all: cutlass.Constexpr,
        ks_full: cutlass.Constexpr,
        ks_part: cutlass.Constexpr,
        k_scalar_full: cutlass.Constexpr,
        k_part_offset: cutlass.Constexpr,
        main_tiles: cutlass.Constexpr,
        tail_tiles: cutlass.Constexpr,
        rank: cutlass.Constexpr,
        hc: cutlass.Constexpr,
        cols_per_cta: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        wid = cute.arch.warp_idx()

        # Fire the dependent-launch signal at CTA start: dependents (the fused
        # up kernel) still gate on griddepcontrol_wait for this grid's output,
        # so their weight streaming fully overlaps this kernel's body.
        cute.arch.griddepcontrol_launch_dependents()

        # Two-phase structure: phase 1 computes every assigned column's dot
        # product in one barrier-free, statically unrolled region so ptxas
        # can keep many columns' loads in flight at once (the whole point of
        # capping the grid); phase 2 then runs the reductions and SiLU
        # epilogues. Per-column FMA order is unchanged -> bit-identical.
        accs = cute.make_rmem_tensor((cols_per_cta, M), cutlass.Float32)
        accs.fill(0.0)

        # K-slice views. tA never depends on the column; tB does.
        if const_expr(k_main_elems > 0):
            gA_main = self._make_k_slice(gA, 0, k_main_elems)
            gB_main = self._make_k_slice(gB, 0, k_main_elems)
            gA_vec = cute.logical_divide(gA_main, (None, main_vec_width))
            gB_vec = cute.logical_divide(gB_main, (None, main_vec_width))
        if const_expr(k_tail_elems > 0):
            gA_tail = self._make_k_slice(gA, k_main_elems, k_tail_elems)
            gB_tail = self._make_k_slice(gB, k_main_elems, k_tail_elems)
            gA_tail_vec = cute.logical_divide(gA_tail, (None, tail_vec_width))
            gB_tail_vec = cute.logical_divide(gB_tail, (None, tail_vec_width))
        if const_expr(ks_full > 0):
            gA_scalar = self._make_k_slice(gA, k_done_all, k_scalar_full)
            gB_scalar = self._make_k_slice(gB, k_done_all, k_scalar_full)
            gA_scalar_vec = cute.logical_divide(gA_scalar, (None, 1))
            gB_scalar_vec = cute.logical_divide(gB_scalar, (None, 1))
        if const_expr(ks_part > 0):
            gA_part = self._make_k_slice(gA, k_part_offset, ks_part)
            gB_part = self._make_k_slice(gB, k_part_offset, ks_part)

        cute.arch.griddepcontrol_wait()

        # Cross-warp reduction scratch, reused per column.
        smem_red_layout = cute.make_layout((M, num_warps), stride=(num_warps, 1))
        smem = cutlass.utils.SmemAllocator()
        sm = smem.allocate_tensor(cutlass.Float32, smem_red_layout, byte_alignment=16)

        # Phase 1: dot products only — no barriers across columns.
        for j in cutlass.range_constexpr(cols_per_cta):
            n_idx = bid * cols_per_cta + j
            if n_idx < N_dim:
                acc = accs[j, None]

                # 128-bit vectorized main loop
                if const_expr(k_main_elems > 0):
                    tA, tB = self._make_thread_vector_slice(
                        gA_vec, gB_vec, tidx, n_idx, threadblock_size
                    )
                    self._vector_fma(acc, tA, tB, M, main_tiles)

                # 64-bit vectorized tail (K remainder after main loop)
                if const_expr(k_tail_elems > 0):
                    tA_t, tB_t = self._make_thread_vector_slice(
                        gA_tail_vec, gB_tail_vec, tidx, n_idx, threadblock_size
                    )
                    self._vector_fma(acc, tA_t, tB_t, M, tail_tiles)

                # Full scalar rounds use CuTe width-1 tiles; KS_PART ragged.
                if const_expr(ks_full > 0):
                    tA_s, tB_s = self._make_thread_vector_slice(
                        gA_scalar_vec,
                        gB_scalar_vec,
                        tidx,
                        n_idx,
                        threadblock_size,
                    )
                    self._vector_fma(acc, tA_s, tB_s, M, ks_full)

                # Only threads below KS_PART load the ragged tail.
                if const_expr(ks_part > 0):
                    if tidx < ks_part:
                        bv2 = gB_part[n_idx, tidx].to(cutlass.Float32)
                        for m in cutlass.range_constexpr(M):
                            acc[m] = acc[m] + gA_part[m, tidx].to(cutlass.Float32) * bv2

        # Phase 2: reductions + fused SiLU epilogues.
        for j in cutlass.range_constexpr(cols_per_cta):
            n_idx = bid * cols_per_cta + j
            if n_idx < N_dim:
                acc = accs[j, None]

                # Intra-warp shuffle reduction
                for m in cutlass.range_constexpr(M):
                    acc[m] = cute.arch.warp_reduction_sum(acc[m])

                with cute.arch.elect_one():
                    for m in cutlass.range_constexpr(M):
                        sm[m, wid] = acc[m]

                # Final reduction, fused mHC SiLU epilogue, and output.
                cute.arch.sync_threads()
                if tidx == 0:
                    for m in cutlass.range_constexpr(M):
                        partials = sm[m, None].load()
                        total = cutlass.Float32(
                            partials.reduce(
                                cute.ReductionOp.ADD,
                                init_val=cutlass.Float32(0.0),
                                reduction_profile=0,
                            )
                        )
                        # Production rounding boundary: GEMM output is
                        # materialized as bf16 before _hc_silu_kernel reads it.
                        x_bf16 = total.to(cutlass.BFloat16)
                        if n_idx < rank:
                            z = x_bf16.to(cutlass.Float32) * (1.0 / hc)
                            gC[m, n_idx] = (z * _sigmoid_f32(z)).to(cutlass.BFloat16)
                        else:
                            gC[m, n_idx] = x_bf16
                # Reuse of the reduction scratch next column must not race the
                # tid0 reads above.
                cute.arch.sync_threads()
