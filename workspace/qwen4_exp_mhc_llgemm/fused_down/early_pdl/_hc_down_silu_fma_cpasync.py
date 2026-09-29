# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""cp.async-pipelined grid-capped down kernel for PDL co-scheduling.

Motivation: the uncapped down FMA kernel needs its 324 CTAs (one per column)
purely for load-latency concurrency — LDG in-flight data is register-bound,
so capping the grid (to free SMs for the PDL-dependent up kernel) starves
the memory pipeline and down stretches ~1.2us per extra column/CTA.

This variant stages weight columns through smem with per-thread cp.async
(16B, .cg), so in-flight data lives in smem, not registers: with S stages a
capped CTA keeps (S-1) columns (~20KB each) in flight at once. The grid is
``ceil(N / cols_per_cta)``; freed SMs let the dependent up kernel
co-schedule at t~1.5us and stream its weights concurrently with down's.

Per-column element assignment and FMA/reduction order are identical to
HcDownSiluFmaEarly (thread t owns K-slice [tile*1024 + t*8, +8)), so output
is bit-identical. Cross-thread smem hazards do not exist: thread t's cp.async
writes are read back only by thread t (same pattern as flash_fwd_combine).

Prototype constraint: K must be a multiple of main_vec_width *
threadblock_size (1024) — the shipped shape K=10240 qualifies; the wrapper
falls back to the plain FMA kernel otherwise.
"""

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.cpasync as cpasync
from cuda.bindings.driver import CUstream
from cutlass import const_expr


def _sigmoid_f32(x):
    return cute.math.rcp(1.0 + cute.math.exp(-x), approx=True)


class HcDownSiluFmaCpAsync:
    """Capped-grid BF16 GEMM with cp.async column pipeline + fused SiLU.

    The compile key is ``(M, K, threadblock_size, rank, hc, cols_per_cta,
    num_stages)``.
    """

    def __init__(
        self,
        k: int,
        cols_per_cta: int,
        num_stages: int = 4,
        threadblock_size: int = 128,
        main_vec_width: int = 8,
        rank: int = 320,
        hc: int = 4,
        min_blocks_per_mp: int = 1,
    ):
        assert k % (main_vec_width * threadblock_size) == 0
        self.k = k
        self.cols_per_cta = cols_per_cta
        self.num_stages = num_stages
        self.threadblock_size = threadblock_size
        self.main_vec_width = main_vec_width
        self.rank = rank
        self.hc = hc
        self.min_blocks_per_mp = min_blocks_per_mp
        self.num_warps = threadblock_size // cute.arch.WARP_SIZE
        self.tiles = k // (main_vec_width * threadblock_size)

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
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            cutlass.BFloat16,
            num_bits_per_copy=self.main_vec_width * 16,
        )
        tiled_copy = cute.make_tiled_copy_tv(
            atom,
            cute.make_layout((1, self.threadblock_size)),
            cute.make_layout((1, self.main_vec_width)),
        )
        self.kernel(
            gA,
            gB,
            gC,
            M,
            N_dim,
            tiled_copy,
            self.main_vec_width,
            self.threadblock_size,
            self.num_warps,
            self.tiles,
            self.rank,
            self.hc,
            self.cols_per_cta,
            self.num_stages,
        ).launch(
            grid=[cute.ceil_div(N_dim, self.cols_per_cta), 1, 1],
            block=[self.threadblock_size, 1, 1],
            smem=self.num_stages * self.k * 2 + M * 4 * self.num_warps,
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
        tiled_copy: cute.TiledCopy,
        main_vec_width: cutlass.Constexpr,
        threadblock_size: cutlass.Constexpr,
        num_warps: cutlass.Constexpr,
        tiles: cutlass.Constexpr,
        rank: cutlass.Constexpr,
        hc: cutlass.Constexpr,
        cols_per_cta: cutlass.Constexpr,
        num_stages: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        wid = cute.arch.warp_idx()

        # Fire the dependent-launch signal at CTA start: dependents (the fused
        # up kernel) still gate on griddepcontrol_wait for this grid's output,
        # so their weight streaming fully overlaps this kernel's body.
        cute.arch.griddepcontrol_launch_dependents()

        smem = cutlass.utils.SmemAllocator()
        sB = smem.allocate_tensor(
            cutlass.BFloat16,
            cute.make_layout((num_stages, self.k), stride=(self.k, 1)),
            byte_alignment=128,
        )
        smem_red_layout = cute.make_layout((M, num_warps), stride=(num_warps, 1))
        sm = smem.allocate_tensor(cutlass.Float32, smem_red_layout, byte_alignment=16)
        # Read view of the same buffer: (VEC, LANE) per (TILE, STAGE), so
        # thread t reads back exactly the 8 elements it copied.
        sB_read = cute.make_tensor(
            sB.iterator,
            cute.make_layout(
                ((main_vec_width, threadblock_size), tiles, num_stages),
                stride=((1, main_vec_width), main_vec_width * threadblock_size, self.k),
            ),
        )

        thr_copy = tiled_copy.get_slice(tidx)
        # gB (N, K) partitioned by the (1, 1024) tile: (VEC, N, TILES).
        tBgB = thr_copy.partition_S(gB)
        # sB (STAGES, K) -> (VEC, STAGES, TILES).
        tBsB = thr_copy.partition_D(sB)

        # A-side thread slices (LDG; L1-resident after the first column).
        gA_vec = cute.logical_divide(gA, (None, main_vec_width))
        tA_full = cute.logical_divide(gA_vec, (None, (None, threadblock_size)))
        tA = tA_full[None, (None, (tidx, None))]  # (M, VEC, TILES)

        # Prologue: issue copies for the first STAGES-1 columns. A commit is
        # issued per slot even when the copy is skipped (ragged CTA), so wait
        # counting below is uniform.
        for s in cutlass.range_constexpr(num_stages - 1):
            if const_expr(s < cols_per_cta):
                n = bid * cols_per_cta + s
                if n < N_dim:
                    for tile in cutlass.range_constexpr(tiles):
                        cute.copy(
                            tiled_copy,
                            tBgB[None, n, tile],
                            tBsB[None, s, tile],
                        )
            cute.arch.cp_async_commit_group()

        cute.arch.griddepcontrol_wait()

        accs = cute.make_rmem_tensor((cols_per_cta, M), cutlass.Float32)
        accs.fill(0.0)

        for j in cutlass.range_constexpr(cols_per_cta):
            n_idx = bid * cols_per_cta + j
            if n_idx < N_dim:
                stage = j % num_stages
                # Column j's group is complete once at most STAGES-2 groups
                # are pending (prologue committed STAGES-1, then one per
                # prior iteration).
                cute.arch.cp_async_wait_group(num_stages - 2)

                # FMA from the staged smem column; A via LDG (L1-hot).
                acc = accs[j, None]
                for tile in cutlass.range_constexpr(tiles):
                    bt = sB_read[(None, tidx), tile, stage]
                    br = cute.make_rmem_tensor_like(bt)
                    cute.autovec_copy(bt, br)
                    br_f32 = br.load().to(cutlass.Float32)
                    for m in cutlass.range_constexpr(M):
                        at = tA[m, None, tile]
                        ar = cute.make_rmem_tensor_like(at)
                        cute.autovec_copy(at, ar)
                        for v in cutlass.range_constexpr(main_vec_width):
                            acc[m] = acc[m] + ar[v].to(cutlass.Float32) * br_f32[v]

                # Issue the next column into the stage being freed. Thread t
                # reuses only smem it copied and already read itself, and the
                # reads precede this copy in program order -> no sync needed.
                if const_expr(j + num_stages - 1 < cols_per_cta):
                    n2 = bid * cols_per_cta + (j + num_stages - 1)
                    if n2 < N_dim:
                        stage2 = (j + num_stages - 1) % num_stages
                        for tile in cutlass.range_constexpr(tiles):
                            cute.copy(
                                tiled_copy,
                                tBgB[None, n2, tile],
                                tBsB[None, stage2, tile],
                            )
                cute.arch.cp_async_commit_group()

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
                # Reduction scratch reuse next column must not race tid0.
                cute.arch.sync_threads()
