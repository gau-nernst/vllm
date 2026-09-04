# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL paged QSA scoring kernel for uniform decode / spec-decode batches.

Per query row r (request r // DQL, q_local = r % DQL; pos = seq_len - DQL +
q_local):
    score[r, c] = sum_h relu(q[r, h] . k[c]) / sqrt(head_dim)   for c < visible[r]
with visible[r] = min((pos + 1) // compress_ratio, seq_len // compress_ratio)
computed by the CALLER (the metadata builder's contract — an input).
over a paged compressed-K cache, writing fp32 logits (columns >= visible are
dead — top-k never reads them) and visible. Uniform batch: DQL is
compile-time and rows are request-major (request == block index); the grid
(split_k, batch) is fixed per CUDA graph.

Restrictions (asserted at the wrapper): num_heads in {2, 4, 8} (head sum
lives in the 4-lane butterfly quartet), head_dim == 128 (swizzle-128B), 0 <
page_size <= 512, DQL <= 8 (BLOCK_Q = 4*DQL <= 32; warm
up 1..1+num_spec), batch <= 65535 (grid dim y cap); Q per-(token, head)
contiguous, K per-block contiguous, 16B-aligned pitches (the vLLM
allocator's guarantee); logits compact fp32 [rows, columns].

Design (adapted from vllm/models/minimax_m3/nvidia/ops/index_decode_score.py):
TMA G2S + mma.sync; one producer warp + NUM_MMA_WARPS = round_up(subpage_size,
32)/32 MMA consumers; 2-stage mbarrier pipeline. The TMA stage height
subpage_size is a dispatch hparam chosen by the wrapper; the mainloop iterates
(page, subpage) units of subpage_size rows, degenerate when subpages == 1. Q
aliases K stage 0; whole-page MMA overhang reads land in a trailing slack
region (masked at the store), subpage ragged tails TMA-zero-fill. Scores
stage in the slack and store with one bulk S2G per unit. PDL:
griddepcontrol_wait in the prologue (off by default).

The kernel uses TMA (SM90+) and is currently validated only for SM100.
"""

import math
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cutlass_dsl import dsl_user_op

from vllm.cute_utils import (
    EVICT_FIRST,
    fence_before_tma_store,
    mma_sync,
    simple_tma_copy,
)


@dsl_user_op
def permute(x: cute.Tensor, dims: tuple[int, ...], *, loc=None, ip=None):
    """Reorder a tensor's modes (CuTe select on the layout).

    Vendored from gn-kernels
    (https://github.com/gau-nernst/gn-kernels, gn_kernels/cutedsl/utils/__init__.py).
    """
    layout = cute.select(x.layout, mode=dims, loc=loc, ip=ip)
    return cute.make_tensor(x.iterator, layout, loc=loc, ip=ip)


def _bulk_s2g(s_stg, g_out, rows: int, dql: int):
    """Bulk-S2G the staging rows with ONE cute.copy: mode 0 = one contiguous
    fp32 row (the per-issue bytes), mode 1 = the DQL rows. `rows` is
    compile-time (fixed by the page_size compile key), so the atom's
    num_bits_per_copy stays constexpr; the composition crops mode 0 to
    `rows`, so the pad rows are unreachable."""
    atom = cute.make_copy_atom(
        cpasync.CopyBulkS2GOp(), Float32, num_bits_per_copy=rows * 32
    )
    # src needs transpose + crop (composition); dst only the transpose
    # (g_out is built with exactly `rows` columns).
    src = cute.composition(s_stg, cute.make_ordered_layout((rows, dql), order=(1, 0)))
    dst = permute(g_out, (1, 0))
    cute.copy(atom, src, dst)


class QsaPagedScoreKernel:
    """TMA producer (warp NUM_MMA_WARPS) + NUM_MMA_WARPS MMA consumer warps,
    one 32-row tile per warp; NUM_MMA_WARPS = round_up(subpage_size, 32) / 32.
    subpage_size (the TMA stage height / page tiling) is a caller-supplied
    dispatch hparam; everything below derives from it."""

    def __init__(
        self,
        dql: int,
        page_size: int,
        split_k: int,
        num_heads: int,
        head_dim: int,
        subpage_size: int,
        use_pdl: bool = False,
    ):
        self.dql = dql
        self.page_size = page_size
        self.split_k = split_k
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.score_scale = 1.0 / math.sqrt(head_dim)
        assert 0 < subpage_size <= page_size
        self.subpage_size = subpage_size
        self.subpages = cute.ceil_div(page_size, self.subpage_size)
        # rows of the last subpage (== subpage_size when it divides evenly)
        self.tail_rows = page_size - self.subpage_size * (self.subpages - 1)
        # MMA rows read per stage: whole 32-row tiles; drives the consumer
        # warp count (one tile per warp) and the smem sizing
        self.num_mma_warps = cute.ceil_div(self.subpage_size, 32)
        self.block_q = num_heads * dql
        # Q columns per request rounded UP to an 8-wide MMA_N tile; padding
        # columns read stale smem and the ql < DQL store mask drops them
        self.block_q_pad = cute.ceil_div(self.block_q, 8) * 8
        # Q must fit in stage 0 (it aliases it): the producer's
        # first K fill of stage 1 does NOT wait for the consumers'
        # Q release, so Q spanning stages would race.
        assert self.block_q_pad <= self.subpage_size, (
            f"Q tile ({self.block_q_pad} rows) exceeds the K stage "
            f"({self.subpage_size} rows)"
        )
        self.use_pdl = use_pdl
        self.num_stages = 2

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [batch * DQL, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, page_size, head_dim]
        page_table: cute.Tensor,  # [batch, max_pages] int32
        visible: cute.Tensor,  # [batch * DQL] int32 (caller-provided)
        logits: cute.Tensor,  # [batch * DQL, columns] fp32
        stream: CUstream,
    ):
        dtype = BFloat16
        DQL = self.dql
        subpage_size = self.subpage_size
        NUM_HEADS = self.num_heads
        HEAD_DIM = self.head_dim
        BLOCK_Q = self.block_q
        assert BLOCK_Q <= 32

        batch = page_table.shape[0]
        # split id is the fastest-varying CTA axis: consecutive CTAs sweep
        # contiguous pages of one request (K locality).
        grid = (self.split_k, batch, 1)
        block = (32 * (self.num_mma_warps + 1), 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        elems = 128 * 8 // dtype.width

        sQ_layout = cute.make_layout(
            (DQL, NUM_HEADS, (elems, HEAD_DIM // elems)),
            stride=(NUM_HEADS * elems, elems, (1, BLOCK_Q * elems)),
        )
        sQ_layout = cute.make_composed_layout(swizzle_128B, 0, sQ_layout)
        Q_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            gQ,
            sQ_layout,
            cta_tiler=(DQL, NUM_HEADS, HEAD_DIM),
        )

        # The TMA box covers exactly one stage: one page, or one 128-row
        # subpage of a larger page (a ragged tail box overhangs the gmem
        # extent and TMA zero-fills it). Stage stride is subpage_size rows
        # (Q aliases stage 0; the fit is asserted in __init__). The kernel
        # reads this layout back from the atom (TmaInfo.smem_layout) instead
        # of rebuilding it.
        sK_layout = cute.make_layout(
            (1, subpage_size, (elems, HEAD_DIM // elems), self.num_stages),
            stride=(
                0,
                elems,
                (1, subpage_size * elems),
                subpage_size * HEAD_DIM,
            ),
        )
        sK_layout = cute.make_composed_layout(swizzle_128B, 0, sK_layout)
        K_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            gK_cache,
            sK_layout,
            cta_tiler=(1, subpage_size, HEAD_DIM),
        )

        self.kernel(
            Q_tma,
            K_tma,
            page_table,
            visible,
            logits,
        ).launch(grid=grid, block=block, stream=stream, use_pdl=self.use_pdl)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        page_table: cute.Tensor,
        visible: cute.Tensor,
        logits: cute.Tensor,
    ):
        split_id, batch_id, _ = cute.arch.block_idx()
        split_k, _, _ = cute.arch.grid_dim()
        warp_id = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_id = cute.arch.lane_idx()

        DQL = self.dql
        page_size = self.page_size
        SUBPAGE_SIZE = self.subpage_size
        SUBPAGES = self.subpages
        TAIL_ROWS = self.tail_rows
        PADDED = self.num_mma_warps * 32
        NUM_MMA_WARPS = self.num_mma_warps
        NUM_HEADS = self.num_heads
        HEAD_DIM = self.head_dim
        dtype = BFloat16
        MMA_N = 8
        num_stages = self.num_stages
        BLOCK_Q = self.block_q
        BLOCK_Q_PAD = self.block_q_pad
        Q_TILES = BLOCK_Q_PAD // MMA_N

        # named barrier allocations
        BAR_MMA = 1

        smem = cutlass.utils.SmemAllocator()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        # One auditable K allocation — every tenant covered by a named term
        # (the stage layout itself comes back from the TMA atom):
        #   k_term:        NUM_STAGES * SUBPAGE_SIZE * HEAD_DIM * 2
        #                  (a stage fits one page/subpage; Q aliases stage 0
        #                  — the fit is asserted in __init__)
        #   overhang_term: (PADDED - SUBPAGE_SIZE) * HEAD_DIM * 2  (MMA reads
        #                  PADDED rows/stage; the last stage's rows
        #                  SUBPAGE_SIZE..PADDED-1 spill here — zero for
        #                  128-row subpage stages)
        #   stage_term:    DQL * PADDED * 4  (bulk_slack score staging)
        k_term = num_stages * SUBPAGE_SIZE * HEAD_DIM * 2
        overhang_term = (PADDED - SUBPAGE_SIZE) * HEAD_DIM * 2
        stage_term = DQL * PADDED * 4
        smem_raw = smem.allocate(
            k_term + max(overhang_term, stage_term), byte_alignment=1024
        )
        smem_ptr = cute.make_ptr(
            dtype,
            smem_raw.toint(),
            cute.AddressSpace.smem,
            assumed_align=1024,
            swizzle_=swizzle_128B,
        )
        sK = cute.make_tensor(smem_ptr, K_tma.smem_layout.outer)[0, None, None, None]
        # alias sQ with sK stage 0
        sQ = cute.make_tensor(smem_ptr, layout=Q_tma.smem_layout.outer)
        # Score staging lives in the slack past the K stages
        # (base = raw base + k_term bytes; k_term is a multiple of 1024).
        sOut = cute.make_tensor(
            cute.make_ptr(
                Float32,
                smem_raw.toint() + k_term,
                cute.AddressSpace.smem,
                assumed_align=1024,
            ),
            cute.make_layout(
                (DQL, PADDED),
                stride=(PADDED, 1),
            ),
        )

        tma_full_mbar = smem.allocate_array(Int64, num_stages)
        tma_empty_mbar = smem.allocate_array(Int64, num_stages)

        # visible is caller-provided; visible_max = max over the request's
        # DQL rows (the last row's by construction — the max is robust)
        visible_max = visible[batch_id * DQL]
        for ql in cutlass.range_constexpr(1, DQL):
            visible_max = cutlass.max(visible_max, visible[batch_id * DQL + ql])
        num_units = cute.ceil_div(visible_max, page_size) * SUBPAGES

        if split_id < num_units:
            if warp_id == 0:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(num_stages):
                        cute.arch.mbarrier_init(tma_full_mbar + i, 1)
                        cute.arch.mbarrier_init(tma_empty_mbar + i, 32 * NUM_MMA_WARPS)
                    cute.arch.mbarrier_init_fence()
            elif warp_id == 1:
                cpasync.prefetch_descriptor(Q_tma.atom)
                cpasync.prefetch_descriptor(K_tma.atom)
            cute.arch.sync_threads()

            if cutlass.const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()

            if warp_id == NUM_MMA_WARPS:
                # TMA producer warp
                tma_stage = 0
                tma_parity = 1

                # tile bid of T = batch * DQL rows is exactly request bid's
                # DQL rows
                # don't need to wait on tma_empty_mbar since this is the 1st TMA load
                gQ_tile = cute.local_tile(
                    Q_tma.tma_tensor,
                    tiler=(DQL, NUM_HEADS, HEAD_DIM),
                    coord=(batch_id, 0, 0),
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        tma_full_mbar, BLOCK_Q * HEAD_DIM * 2
                    )
                simple_tma_copy(Q_tma.atom, gQ_tile, sQ, tma_full_mbar)

                tma_stage = (tma_stage + 1) % num_stages
                if tma_stage == 0:
                    tma_parity ^= 1

                for unit in range(split_id, num_units, split_k):
                    page_id = page_table[batch_id, unit // SUBPAGES]
                    # the sub-th subpage_size-row box of the page; a ragged
                    # tail box overhangs the gmem extent and TMA zero-fills
                    gK_tile = cute.local_tile(
                        K_tma.tma_tensor[page_id, None, None],
                        (SUBPAGE_SIZE, HEAD_DIM),
                        (unit % SUBPAGES, 0),
                    )
                    k_mbar = tma_full_mbar + tma_stage

                    cute.arch.mbarrier_wait(tma_empty_mbar + tma_stage, tma_parity)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            k_mbar, SUBPAGE_SIZE * HEAD_DIM * 2
                        )
                    simple_tma_copy(
                        K_tma.atom,
                        gK_tile,
                        sK[None, None, tma_stage],
                        k_mbar,
                        cache_policy=EVICT_FIRST,
                    )

                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1
            else:
                # NUM_MMA_WARPS MMA consumer warps, one 32-row K tile each: warp w
                # owns tile w. Lane mapping: row lane%16 of each 16-row
                # ldmatrix group, 16B column half lane//16.
                elems = 128 // dtype.width  # 16B
                MMA_K = 32 * 8 // dtype.width  # 32B

                sK_warp = cute.local_tile(
                    sK, (32, HEAD_DIM, num_stages), (warp_id, 0, 0)
                )
                sK_ldsm = cute.zipped_divide(
                    sK_warp, (16, cute.make_layout((elems, 2)), 1)
                )
                sK_ldsm = sK_ldsm[(lane_id % 16, (None, lane_id // 16), 0), None]
                # ldmatrix view of sQ: the smem box lands heads-innermost
                # (physical N col = ql * NUM_HEADS + head), so the logical N
                # mode is a plain flat range, padded from BLOCK_Q to a full
                # MMA_N tile (DQL=1). The padding columns read aliased
                # in-allocation smem and the epilogue's ql < DQL store mask
                # never stores them.
                elems_q = 128 * 8 // dtype.width  # 64: one 128B swizzle atom
                sQ_mma = cute.make_tensor(
                    sQ.iterator,
                    cute.make_layout(
                        (BLOCK_Q_PAD, (elems_q, HEAD_DIM // elems_q)),
                        stride=(elems_q, (1, BLOCK_Q * elems_q)),
                    ),
                )
                sQ_ldsm = cute.zipped_divide(
                    sQ_mma, (MMA_N, cute.make_layout((elems, 4)))
                )
                sQ_ldsm = sQ_ldsm[(lane_id % MMA_N, (None, lane_id // 8)), None]

                ldsm_op = warp.LdMatrix8x8x16bOp(num_matrices=4)
                ldsm_atom = cute.make_copy_atom(ldsm_op, dtype)

                rQ = cute.make_rmem_tensor(
                    ((elems // 2, 2), HEAD_DIM // (MMA_K * 2), Q_TILES), dtype
                )
                rK = cute.make_rmem_tensor((elems, 2, HEAD_DIM // MMA_K), dtype)
                rC = cute.make_rmem_tensor((4, 2, Q_TILES), Float32)

                if warp_id == 0:
                    cute.arch.mbarrier_wait(tma_full_mbar, 0)
                cute.arch.barrier(
                    barrier_id=BAR_MMA, number_of_threads=32 * NUM_MMA_WARPS
                )
                for q in cutlass.range_constexpr(Q_TILES):
                    cute.copy(ldsm_atom, sQ_ldsm[None, (q, None)], rQ[None, None, q])
                # release-safety barrier: see the mainloop release site below
                cute.arch.barrier(
                    barrier_id=BAR_MMA, number_of_threads=32 * NUM_MMA_WARPS
                )
                cute.arch.mbarrier_arrive(tma_empty_mbar)

                # The K pipeline starts on stage 1: stage 0 holds the Q box.
                tma_stage = 1 % num_stages
                tma_parity = 0
                if tma_stage == 0:
                    tma_parity ^= 1

                row8 = lane_id // 4
                m4 = lane_id % 4
                for unit in range(split_id, num_units, split_k):
                    if warp_id == 0:
                        cute.arch.mbarrier_wait(tma_full_mbar + tma_stage, tma_parity)
                    cute.arch.barrier(
                        barrier_id=BAR_MMA, number_of_threads=32 * NUM_MMA_WARPS
                    )
                    page_idx = unit // SUBPAGES
                    sub = unit % SUBPAGES
                    unit_base = page_idx * page_size + sub * SUBPAGE_SIZE
                    tile = warp_id  # this warp's one 32-row tile of the stage

                    # MMA block.
                    rC.fill(0.0)
                    for k in cutlass.range_constexpr(HEAD_DIM // MMA_K):
                        cute.copy(
                            ldsm_atom,
                            sK_ldsm[None, (None, k, tma_stage)],
                            rK[None, None, k],
                        )
                        for m in cutlass.range_constexpr(2):
                            for n in cutlass.range_constexpr(Q_TILES):
                                rC[None, m, n] = mma_sync(
                                    rK[None, m, k],
                                    rQ[(None, k % 2), k // 2, n],
                                    rC[None, m, n],
                                )

                    # Restage gate (NOT the K-stage release): single-buffered
                    # staging may be rewritten only after the prior page's S2G
                    # drains; the shared bar.sync below releases this gate to
                    # all consumer warps.
                    if warp_id == 0:
                        with cute.arch.elect_one():
                            cute.arch.cp_async_bulk_wait_group(0, read=True)

                    # The ONE K-stage release. mbarrier.arrive is
                    # release-cta and orders this thread's prior ldmatrix
                    # reads before the TMA refill may observe the free stage;
                    # the bar.sync before it is required on this hardware
                    # (arrive does not reliably observe in-flight ldmatrix —
                    # see tma_release_race_test.py). Every consumer thread
                    # arrives (count at mbarrier init).
                    cute.arch.barrier(
                        barrier_id=BAR_MMA,
                        number_of_threads=32 * NUM_MMA_WARPS,
                    )
                    cute.arch.mbarrier_arrive(tma_empty_mbar + tma_stage)

                    # Padded-row-strided staging (round_up(subpage_size, 32)):
                    # unpredicated pad writes land in the per-row pad; the
                    # S2G copies only the valid rows. (sOut: pre-loop.)
                    # relu + head-sum + stage this warp's 32-key tile.
                    # ql-major N packing (col = ql * num_heads + head): rC
                    # linear index q*8 + i*2 + j is key row tile*32 + i*8 +
                    # lane//4, N column q*8 + (lane%4)*2 + j. The head sum
                    # stays inside the 4-lane butterfly quartet (m4=lane%4),
                    # which holds 8/num_heads ql-groups: num_heads=2 — one
                    # lane per ql (no shuffle); 4 — lane pairs (xor-1); 8 —
                    # the whole quartet (xor-1 then xor-2). The group's
                    # lowest lane owns the store. Padded N columns
                    # (ql >= DQL) are masked by ql_q < DQL.
                    local0 = tile * 32 + row8
                    for q in cutlass.range_constexpr(Q_TILES):
                        ql_q = (q * 8 + m4 * 2) // NUM_HEADS
                        for i in cutlass.range_constexpr(4):
                            p = cute.arch.fmax(
                                rC[q * 8 + i * 2], Float32(0.0)
                            ) + cute.arch.fmax(rC[q * 8 + i * 2 + 1], Float32(0.0))
                            if cutlass.const_expr(NUM_HEADS >= 4):
                                p = p + cute.arch.shuffle_sync_bfly(p, offset=1)
                            if cutlass.const_expr(NUM_HEADS == 8):
                                p = p + cute.arch.shuffle_sync_bfly(p, offset=2)
                            local = local0 + i * 8
                            if m4 % max(1, NUM_HEADS // 2) == 0 and ql_q < DQL:
                                sOut[ql_q, local] = p * self.score_scale
                    cute.arch.barrier(
                        barrier_id=BAR_MMA,
                        number_of_threads=32 * NUM_MMA_WARPS,
                    )
                    if warp_id == 0:
                        with cute.arch.elect_one():
                            fence_before_tma_store()
                            # this unit's [DQL, SUBPAGE_SIZE] gmem tile
                            gOut = cute.local_tile(
                                cute.domain_offset((batch_id * DQL, unit_base), logits),
                                (DQL, SUBPAGE_SIZE),
                                (0, 0),
                            )
                            if cutlass.const_expr(TAIL_ROWS == SUBPAGE_SIZE):
                                _bulk_s2g(sOut, gOut, SUBPAGE_SIZE, DQL)
                            elif sub == SUBPAGES - 1:
                                # ragged tail: row count is fixed by the
                                # page_size compile key; the narrow tile
                                # needs no crop on the dst side
                                gOut_tail = cute.local_tile(
                                    cute.domain_offset(
                                        (batch_id * DQL, unit_base), logits
                                    ),
                                    (DQL, TAIL_ROWS),
                                    (0, 0),
                                )
                                _bulk_s2g(sOut, gOut_tail, TAIL_ROWS, DQL)
                            else:
                                _bulk_s2g(sOut, gOut, SUBPAGE_SIZE, DQL)
                            cute.arch.cp_async_bulk_commit_group()
                    tma_stage = (tma_stage + 1) % num_stages
                    if tma_stage == 0:
                        tma_parity ^= 1

    @cache
    @staticmethod
    def compile(
        dql: int,
        page_size: int,
        split_k: int,
        num_heads: int,
        head_dim: int,
        subpage_size: int,
        use_pdl: bool = False,
    ):
        def _fake(dtype, shape, div):
            stride = tuple(
                cute.sym_int64(divisibility=div) if i < len(shape) - 1 else 1
                for i in range(len(shape))
            )
            return cute.runtime.make_fake_tensor(
                dtype,
                shape,
                stride=stride,
                assumed_align=max(div * dtype.width // 8, 1),
            )

        bs = cute.sym_int()
        total_tokens = cute.sym_int()
        q = _fake(BFloat16, (cute.sym_int(), num_heads, head_dim), 8)
        k_cache = _fake(BFloat16, (cute.sym_int(), page_size, head_dim), 8)
        page_table = _fake(Int32, (bs, cute.sym_int()), 1)
        visible = _fake(Int32, (total_tokens,), 1)
        logits = _fake(Float32, (total_tokens, cute.sym_int()), 4)
        kernel = QsaPagedScoreKernel(
            dql,
            page_size,
            split_k,
            num_heads,
            head_dim,
            subpage_size,
            use_pdl,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k_cache,
            page_table,
            visible,
            logits,
            stream,
            options="--enable-tvm-ffi",
        )


def _default_subpage_size(page_size: int) -> int:
    # Dispatch policy: whole-page stages up to 224 rows (2 stages still fit
    # 2 CTA/SM); 128-row subpage stages above (two whole-page stages would
    # cliff to 1 CTA/SM there).
    return page_size if page_size <= 224 else 128


def _auto_split_k(batch: int) -> int:
    # Grid is (split_k, batch): both factors are capture-time constants under
    # uniform-batch CUDA graphs. Small batches want maximum split parallelism;
    # large batches want to avoid the empty-CTA tail. This can NOT key on
    # context length: under CUDA-graph capture the kernel is planned against
    # padded shapes (max batch, max_model_len ctx), so context-dependent
    # dispatch is graph-incompatible.
    if batch <= 4:
        return 256
    if batch <= 16:
        return 64
    return 32


def warmup_qsa_paged_score_cutedsl(
    num_heads: int,
    head_dim: int,
    page_size: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Compile every dispatch-reachable (dql, split_k) specialization.

    Compile-only (fake tensors, no launch); runtime launches reuse the cache.
    """
    subpage_size = _default_subpage_size(page_size)
    profiles: list[tuple[int, int]] = []
    for dql in range(1, max_decode_query_len + 1):
        if not qsa_score_cutedsl_config_supported(num_heads, head_dim, page_size, dql):
            continue
        max_batch = min(max_num_reqs, max_num_batched_tokens // dql)
        split_ks = [256]
        if max_batch > 4:
            split_ks.append(64)
        if max_batch > 16:
            split_ks.append(32)
        for split_k in split_ks:
            QsaPagedScoreKernel.compile(
                dql, page_size, split_k, num_heads, head_dim, subpage_size
            )
            profiles.append((dql, split_k))
    return tuple(profiles)


def qsa_score_cutedsl_config_supported(
    num_heads: int,
    head_dim: int,
    page_size: int,
    dql: int,
) -> bool:
    """Static dispatch gate: config values the CuteDSL score kernel supports."""
    if num_heads not in (2, 4, 8) or head_dim != 128:
        return False
    if not 0 < page_size <= 512:
        return False
    block_q_pad = -(-num_heads * dql // 8) * 8
    return (
        0 < dql <= 8
        and block_q_pad <= 32
        and (block_q_pad <= _default_subpage_size(page_size))
    )


def qsa_score_cutedsl_supported(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    dql: int,
) -> bool:
    """Full dispatch gate: static config plus the runtime tensor contract."""
    num_heads, head_dim = q.shape[1], q.shape[2]
    page_size = k_cache.shape[1]
    if not qsa_score_cutedsl_config_supported(num_heads, head_dim, page_size, dql):
        return False
    if page_table.shape[0] > 65535 or q.shape[0] != page_table.shape[0] * dql:
        return False
    return bool(
        q.dtype == torch.bfloat16
        and k_cache.dtype == torch.bfloat16
        and q.stride(2) == 1
        and q.stride(1) == head_dim
        and q.stride(0) % 8 == 0
        and q.data_ptr() % 16 == 0
        and k_cache.ndim == 4
        and k_cache.shape[2] == 1
        and k_cache.stride(3) == 1
        and k_cache.stride(1) == head_dim
        and k_cache.stride(0) % 8 == 0
        and k_cache.data_ptr() % 16 == 0
    )


def qsa_paged_score_cutedsl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible: torch.Tensor,
    logits: torch.Tensor,
    dql: int,
    split_k: int | None = None,
    subpage_size: int | None = None,
    use_pdl: bool = False,
) -> None:
    """Score one uniform batch into logits (visible is a caller input).

    Scores stage in the K allocation's trailing slack and store with one
    bulk S2G per page. Interface: q needs only per-(token, head) contiguity
    ([T, 4, 128] with strides (*, 128, 1); the token pitch is a runtime
    tensor-map value); k_cache must be page-contiguous ([pages, page_size, 1,
    128] with row strides (128, 1); the page pitch is likewise a runtime
    tensor-map value).
    """
    num_heads, head_dim = q.shape[1], q.shape[2]
    batch = page_table.shape[0]
    page_size = k_cache.shape[1]
    dql = int(dql)
    if split_k is None:
        split_k = _auto_split_k(batch)
    if subpage_size is None:
        subpage_size = _default_subpage_size(page_size)
    # Split-fastest grid: batch rides on grid dim y (cap 65535; x allows
    # 2**31-1), so fail loudly past it.
    if batch > 65535:
        raise ValueError(f"QSA batch {batch} exceeds the grid-dim-y cap 65535")
    assert q.shape[0] == batch * dql
    assert 0 < page_size <= 512, f"QSA page_size must be in (0, 512], got {page_size}"
    # Interface contract: Q needs only per-(token, head) contiguity
    # ([T, 4, 128] with strides (*, 128, 1); the token pitch is a runtime
    # tensor-map value — e.g. a row-strided view of a fused QK projection
    # buffer). Within a page the K rows are contiguous ([page_size, 128]
    # with strides (128, 1)); the page (block) pitch is arbitrary —
    # _reshape_attention_kv_cache produces non-compact block strides — so it
    # is a runtime tensor-map pitch too.
    if not (
        q.dtype == torch.bfloat16
        and q.shape[1] in (2, 4, 8)
        and q.shape[2] == 128
        and q.stride(2) == 1
        and q.stride(1) == head_dim
        and q.stride(0) % 8 == 0  # 16B-aligned token pitch
        and q.data_ptr() % 16 == 0
    ):
        raise ValueError(
            "QSA CuteDSL kernel requires bf16, num_heads in {2, 4, 8} (power "
            "of 2: the head reduction lives in the 4-lane butterfly quartet) "
            "and head_dim 128 (swizzle-128B geometry) with per-(token, "
            f"head) contiguity, got {q.dtype} {num_heads}x{head_dim} strides "
            f"{q.stride()}"
        )
    if not (
        k_cache.dtype == torch.bfloat16
        and k_cache.ndim == 4
        and k_cache.shape[2] == 1
        and k_cache.stride(3) == 1
        and k_cache.stride(1) == head_dim
        and k_cache.stride(0) % 8 == 0  # 16B-aligned block pitch
        and k_cache.data_ptr() % 16 == 0
    ):
        raise ValueError(
            f"QSA CuteDSL kernel requires bf16 page-contiguous k_cache, got "
            f"{k_cache.dtype} shape {k_cache.shape} strides {k_cache.stride()}"
        )
    rows = batch * dql
    if not (
        logits.dtype == torch.float32
        and logits.shape[0] == rows
        and logits.stride(1) == 1
        and logits.stride(0) == logits.shape[1]
        and logits.shape[1] % page_size == 0
    ):
        raise ValueError(
            "QSA CuteDSL kernel requires fp32 contiguous logits of shape "
            "[batch * dql, columns] with columns a multiple of page_size "
            "(the store paths tile columns by page), got shape "
            f"{logits.shape} dtype {logits.dtype} strides {logits.stride()}"
        )
    if not (
        visible.dtype == torch.int32
        and visible.shape == (rows,)
        and visible.stride(0) == 1
    ):
        raise ValueError(
            "QSA CuteDSL kernel requires int32 contiguous visible of shape "
            f"[batch * dql], got shape {visible.shape} dtype {visible.dtype}"
        )
    if page_table.dtype != torch.int32 or page_table.stride(1) != 1:
        raise ValueError(
            "QSA CuteDSL kernel requires an int32 row-contiguous page_table, "
            f"got shape {page_table.shape} dtype {page_table.dtype} strides "
            f"{page_table.stride()}"
        )
    kernel = QsaPagedScoreKernel.compile(
        dql,
        page_size,
        split_k,
        num_heads,
        head_dim,
        subpage_size,
        use_pdl,
    )
    kernel(
        q,
        k_cache[:, :, 0, :],
        page_table,
        visible,
        logits,
    )
