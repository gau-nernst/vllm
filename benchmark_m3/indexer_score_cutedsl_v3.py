# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import cache

import cutlass
import torch
from cuda.bindings.driver import CUstream
from cutlass import BFloat16, Float32, Int32, Int64, cute
from cutlass.cute.nvgpu import cpasync
from quack.compile_utils import make_fake_tensor

from vllm.cute_utils import _tcgen05, simple_tma_copy


class Sm100MSAIndexerScoreKernelV3:
    BLOCK_Q = 128
    BLOCK_K = 128
    GROUP_K = 32
    num_stages = 3
    num_q_stages = 2

    def __init__(self, num_heads: int, head_dim: int = 128):
        self.num_heads = num_heads
        self.head_dim = head_dim

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [total_tokens, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, page_size, head_dim]
        block_table: cute.Tensor,  # [num_seqs, max_pages]
        work_tiles: cute.Tensor,  # [num_work_tiles, 4]
        work_offsets: cute.Tensor,  # [num_ctas+1]
        score: cute.Tensor,  # [num_heads, total_tokens, max_pages]
        cu_seqlens_q: cute.Tensor,  # [num_seqs+1]
        context_lens: cute.Tensor,  # [num_seqs]
        num_sms: Int32,
        stream: CUstream,
    ):
        BLOCK_Q = self.BLOCK_Q
        BLOCK_K = self.BLOCK_K
        head_dim = self.head_dim

        num_heads = self.num_heads
        grid_x = max(1, num_sms // num_heads)
        grid = (num_heads, grid_x, 1)
        block = (32 * 6, 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        sQ_layout = cute.make_layout(
            (BLOCK_Q, 1, (64, self.head_dim // 64), self.num_q_stages),
            stride=(64, 0, (1, BLOCK_Q * 64), BLOCK_Q * head_dim),
        )
        sQ_layout = cute.make_composed_layout(swizzle_128B, 0, sQ_layout)
        Q_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gQ, (None, None, 64)),
            sQ_layout,
            cta_tiler=(BLOCK_Q, 1, head_dim),
        )

        num_stages = self.num_stages
        sK_layout = cute.make_layout(
            (1, BLOCK_K, (64, self.head_dim // 64), num_stages),
            stride=(0, 64, (1, BLOCK_K * 64), BLOCK_K * head_dim),
        )
        sK_layout = cute.make_composed_layout(swizzle_128B, 0, sK_layout)
        K_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gK_cache, (None, None, 64)),
            sK_layout,
            cta_tiler=(1, BLOCK_K, head_dim),
        )

        self.kernel(
            Q_tma,
            K_tma,
            block_table,
            work_tiles,
            work_offsets,
            score,
            cu_seqlens_q,
            context_lens,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        block_table: cute.Tensor,  # [num_seqs, max_pages]
        work_tiles: cute.Tensor,  # [num_work_tiles, 4]
        work_offsets: cute.Tensor,  # [num_ctas+1]
        score: cute.Tensor,  # [num_heads, total_tokens, max_pages]
        cu_seqlens_q: cute.Tensor,  # [num_seqs+1]
        context_lens: cute.Tensor,  # [num_seqs]
    ):
        tid, _, _ = cute.arch.thread_idx()
        head_id, bid, _ = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BLOCK_Q = self.BLOCK_Q
        BLOCK_K = self.BLOCK_K
        num_stages = self.num_stages
        num_q_stages = self.num_q_stages
        head_dim = self.head_dim

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        smem = cutlass.utils.SmemAllocator()
        sQ = allocate_tensor(smem, BFloat16, Q_tma.smem_layout)[None, 0, None, None]
        sK = allocate_tensor(smem, BFloat16, K_tma.smem_layout)[0, None, None, None]

        tma_mbar = smem.allocate_array(Int64, num_stages)
        mma_mbar = smem.allocate_array(Int64, num_stages)
        epi_mbar = smem.allocate_array(Int64, num_stages)
        q_empty_mbar = smem.allocate_array(Int64, num_q_stages)
        taddr = smem.allocate_array(Int32, 1)
        assert BLOCK_K * num_stages <= 512

        if warp_id == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(num_stages):
                    cute.arch.mbarrier_init(tma_mbar + i, 1)
                    cute.arch.mbarrier_init(mma_mbar + i, 1)
                    cute.arch.mbarrier_init(epi_mbar + i, 128)
                for i in cutlass.range_constexpr(num_q_stages):
                    cute.arch.mbarrier_init(q_empty_mbar + i, 1)
                cute.arch.mbarrier_init_fence()
        elif warp_id == 1:
            cpasync.prefetch_descriptor(Q_tma.atom)
            cpasync.prefetch_descriptor(K_tma.atom)
        cute.arch.sync_threads()

        work_begin = work_offsets[bid]
        work_end = work_offsets[bid + 1]

        if work_begin < work_end:
            if warp_id == 5:
                stage_id = 0
                parity = 1
                q_stage_id = 0
                q_parity = 1

                for work_tile_id in cutlass.range(work_begin, work_end, unroll=1):
                    req_id = work_tiles[work_tile_id, 0]
                    q_tile_id = work_tiles[work_tile_id, 1]
                    logical_k_block_start = work_tiles[work_tile_id, 2]
                    group_len = work_tiles[work_tile_id, 3]
                    bos = cu_seqlens_q[req_id]

                    logical_k_block_id = logical_k_block_start
                    page_id = block_table[req_id, logical_k_block_id]

                    q_tile = cute.local_tile(
                        cute.domain_offset(
                            (bos, 0),
                            Q_tma.tma_tensor[None, head_id, None],
                        ),
                        tiler=(BLOCK_Q, head_dim),
                        coord=(q_tile_id, 0),
                    )
                    k_tile = K_tma.tma_tensor[page_id, None, None]
                    mbar = tma_mbar + stage_id

                    cute.arch.mbarrier_wait(q_empty_mbar + q_stage_id, q_parity)
                    cute.arch.mbarrier_wait(mma_mbar + stage_id, parity)
                    simple_tma_copy(
                        Q_tma.atom, q_tile, sQ[None, None, q_stage_id], mbar
                    )
                    simple_tma_copy(K_tma.atom, k_tile, sK[None, None, stage_id], mbar)

                    with cute.arch.elect_one():
                        qk_bytes = (BLOCK_Q + BLOCK_K) * head_dim * 2
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar, qk_bytes)

                    q_stage_id = (q_stage_id + 1) % num_q_stages
                    if q_stage_id == 0:
                        q_parity ^= 1

                    stage_id = (stage_id + 1) % num_stages
                    if stage_id == 0:
                        parity ^= 1

                    for group_offset in cutlass.range(1, group_len, unroll=1):
                        logical_k_block_id = logical_k_block_start + group_offset
                        page_id = block_table[req_id, logical_k_block_id]
                        k_tile = K_tma.tma_tensor[page_id, None, None]
                        mbar = tma_mbar + stage_id

                        cute.arch.mbarrier_wait(mma_mbar + stage_id, parity)

                        with cute.arch.elect_one():
                            k_bytes = BLOCK_K * head_dim * 2
                            cute.arch.mbarrier_arrive_and_expect_tx(mbar, k_bytes)
                        simple_tma_copy(
                            K_tma.atom, k_tile, sK[None, None, stage_id], mbar
                        )

                        stage_id = (stage_id + 1) % num_stages
                        if stage_id == 0:
                            parity ^= 1

            elif warp_id == 4:
                qdesc_template = _tcgen05.make_sdesc_128B_swizzle(BLOCK_Q * 128)
                kdesc_template = _tcgen05.make_sdesc_128B_swizzle(BLOCK_K * 128)
                idesc = _tcgen05.make_bf16_idesc(BLOCK_Q, BLOCK_K)

                cute.arch.barrier(barrier_id=2, number_of_threads=5 * 32)

                stage_id = 0
                parity = 0
                q_stage_id = 0
                for work_tile_id in cutlass.range(work_begin, work_end, unroll=1):
                    group_len = work_tiles[work_tile_id, 3]

                    q_addr = sQ[None, None, q_stage_id].iterator.toint()
                    qdesc_base = qdesc_template | (q_addr >> 4)

                    for _ in cutlass.range(group_len, unroll=1):
                        cute.arch.mbarrier_wait(epi_mbar + stage_id, parity ^ 1)

                        k_addr = sK[None, None, stage_id].iterator.toint()
                        kdesc_base = kdesc_template | (k_addr >> 4)
                        acc_tmem = BLOCK_K * stage_id

                        cute.arch.mbarrier_wait(tma_mbar + stage_id, parity)
                        _tcgen05.fence_after_thread_sync()

                        for i in cutlass.range_constexpr(head_dim // 64):
                            for j in cutlass.range_constexpr(64 // 16):
                                qdesc = qdesc_base | ((i * BLOCK_Q * 128 + j * 32) >> 4)
                                kdesc = kdesc_base | ((i * BLOCK_K * 128 + j * 32) >> 4)
                                _tcgen05.mma_f16(
                                    acc_tmem,
                                    qdesc,
                                    kdesc,
                                    idesc,
                                    (i > 0) or (j > 0),
                                )
                        _tcgen05.commit(mma_mbar + stage_id)
                        stage_id = (stage_id + 1) % num_stages
                        if stage_id == 0:
                            parity ^= 1

                    _tcgen05.commit(q_empty_mbar + q_stage_id)
                    q_stage_id = (q_stage_id + 1) % num_q_stages

            else:
                if warp_id == 0:
                    _tcgen05.alloc(taddr)
                cute.arch.barrier(barrier_id=2, number_of_threads=5 * 32)

                WIDTH = 64

                stage_id = 0
                parity = 0
                for work_tile_id in cutlass.range(work_begin, work_end, unroll=1):
                    req_id = work_tiles[work_tile_id, 0]
                    q_tile_id = work_tiles[work_tile_id, 1]
                    logical_k_block_start = work_tiles[work_tile_id, 2]
                    group_len = work_tiles[work_tile_id, 3]

                    bos = cu_seqlens_q[req_id]
                    q_start = bos + q_tile_id * BLOCK_Q
                    eos = cu_seqlens_q[req_id + 1]
                    q_len = min(eos - q_start, BLOCK_Q)
                    context_len = context_lens[req_id]
                    local_q_base = q_tile_id * BLOCK_Q
                    q_tile_is_full = q_len == BLOCK_Q

                    q_pos = tid
                    for group_offset in cutlass.range(group_len, unroll=1):
                        logical_k_block_id = logical_k_block_start + group_offset
                        k_tile_is_full = (
                            logical_k_block_id + 1
                        ) * BLOCK_K <= context_len + local_q_base + 1
                        tile_is_full = q_tile_is_full and k_tile_is_full

                        if warp_id == 0:
                            cute.arch.mbarrier_wait(mma_mbar + stage_id, parity)
                        cute.arch.barrier(barrier_id=1, number_of_threads=128)
                        _tcgen05.fence_after_thread_sync()

                        max_score = cute.make_rmem_tensor(4, Float32)
                        max_score.fill(float("-inf"))

                        if tile_is_full:
                            for i in cutlass.range_constexpr(BLOCK_K // WIDTH):
                                tcol = BLOCK_K * stage_id + i * WIDTH
                                qk = _tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH)
                                _tcgen05.wait_ld()

                                if cutlass.const_expr(i == BLOCK_K // WIDTH - 1):
                                    cute.arch.mbarrier_arrive(epi_mbar + stage_id)
                                for j in cutlass.range_constexpr(WIDTH):
                                    max_score[j % 4] = cute.arch.fmax(
                                        max_score[j % 4], qk[j]
                                    )

                            max_score[0] = cute.arch.fmax(max_score[0], max_score[1])
                            max_score[2] = cute.arch.fmax(max_score[2], max_score[3])
                            max_score[0] = cute.arch.fmax(max_score[0], max_score[2])

                            score[head_id, q_start + q_pos, logical_k_block_id] = (
                                max_score[0]
                            )

                        else:
                            local_k_len = min(
                                context_len
                                + local_q_base
                                + q_pos
                                + 1
                                - logical_k_block_id * BLOCK_K,
                                BLOCK_K,
                            )

                            for i in cutlass.range_constexpr(BLOCK_K // WIDTH):
                                tcol = BLOCK_K * stage_id + i * WIDTH
                                qk = _tcgen05.ld(warp_id * 32, tcol, "32x32b", WIDTH)
                                _tcgen05.wait_ld()

                                if cutlass.const_expr(i == BLOCK_K // WIDTH - 1):
                                    cute.arch.mbarrier_arrive(epi_mbar + stage_id)
                                for j in range(WIDTH):
                                    if j < local_k_len - i * WIDTH:
                                        max_score[j % 4] = cute.arch.fmax(
                                            max_score[j % 4], qk[j]
                                        )

                            if q_pos < q_len:
                                max_score[0] = cute.arch.fmax(
                                    max_score[0], max_score[1]
                                )
                                max_score[2] = cute.arch.fmax(
                                    max_score[2], max_score[3]
                                )
                                max_score[0] = cute.arch.fmax(
                                    max_score[0], max_score[2]
                                )
                                score[
                                    head_id,
                                    q_start + q_pos,
                                    logical_k_block_id,
                                ] = max_score[0]

                        stage_id = (stage_id + 1) % num_stages
                        if stage_id == 0:
                            parity ^= 1

                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile(
        num_heads: int,
        head_dim: int = 128,
    ):
        total_tokens = cute.sym_int()
        num_work_tiles = cute.sym_int()
        num_seqs = cute.sym_int()

        q = make_fake_tensor(
            BFloat16, (total_tokens, num_heads, head_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            BFloat16,
            (cute.sym_int(), Sm100MSAIndexerScoreKernelV3.BLOCK_K, head_dim),
            divisibility=16,
        )
        block_table = make_fake_tensor(
            Int32, (num_seqs, cute.sym_int()), divisibility=1
        )
        work_tiles = make_fake_tensor(Int32, (num_work_tiles, 4), divisibility=4)
        work_offsets = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
        score = make_fake_tensor(
            Float32, (num_heads, total_tokens, cute.sym_int()), divisibility=4
        )
        cu_seqlens_q = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
        context_lens = make_fake_tensor(Int32, (num_seqs,), divisibility=1)

        kernel = Sm100MSAIndexerScoreKernelV3(num_heads, head_dim)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k_cache,
            block_table,
            work_tiles,
            work_offsets,
            score,
            cu_seqlens_q,
            context_lens,
            Int32(128),
            stream,
            options="--enable-tvm-ffi",
        )


def prepare_indexer_score_cutedsl_v3_work_tiles(
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    num_heads: int = 1,
    num_sms: int = 148,
) -> tuple[torch.Tensor, torch.Tensor]:
    cu_seqlens_cpu = cu_seqlens_q.cpu().tolist()
    context_lens_cpu = context_lens.cpu().tolist()

    q_tiles: list[tuple[int, int, int]] = []
    for req_id, context_len in enumerate(context_lens_cpu):
        q_len = cu_seqlens_cpu[req_id + 1] - cu_seqlens_cpu[req_id]
        num_q_tiles = (q_len + Sm100MSAIndexerScoreKernelV3.BLOCK_Q - 1) // (
            Sm100MSAIndexerScoreKernelV3.BLOCK_Q
        )
        for q_tile_id in range(num_q_tiles):
            q_tile_end = min(
                (q_tile_id + 1) * Sm100MSAIndexerScoreKernelV3.BLOCK_Q,
                q_len,
            )
            num_k_blocks = (
                context_len + q_tile_end + Sm100MSAIndexerScoreKernelV3.BLOCK_K - 1
            ) // Sm100MSAIndexerScoreKernelV3.BLOCK_K
            q_tiles.append((req_id, q_tile_id, num_k_blocks))

    grid_x = max(1, num_sms // num_heads)
    total_qk_tiles = sum(num_k_blocks for _, _, num_k_blocks in q_tiles)
    if total_qk_tiles == 0:
        work_tiles = torch.empty((0, 4), device=block_table.device, dtype=torch.int32)
        work_offsets = torch.zeros(
            grid_x + 1, device=block_table.device, dtype=torch.int32
        )
        return work_tiles, work_offsets

    work_tiles_list: list[tuple[int, int, int, int]] = []
    work_offsets_list = [0]
    tile_id = 0
    k_start = 0
    assigned_qk_tiles = 0
    for cta_id in range(grid_x):
        target_end = ((cta_id + 1) * total_qk_tiles + grid_x - 1) // grid_x
        remaining = target_end - assigned_qk_tiles
        while remaining > 0 and tile_id < len(q_tiles):
            req_id, q_tile_id, num_k_blocks = q_tiles[tile_id]
            group_len = min(remaining, num_k_blocks - k_start)
            work_tiles_list.append((req_id, q_tile_id, k_start, group_len))
            remaining -= group_len
            assigned_qk_tiles += group_len
            k_start += group_len
            if k_start == num_k_blocks:
                tile_id += 1
                k_start = 0
        work_offsets_list.append(len(work_tiles_list))

    while len(work_offsets_list) < grid_x + 1:
        work_offsets_list.append(len(work_tiles_list))

    work_tiles = torch.tensor(
        work_tiles_list, device=block_table.device, dtype=torch.int32
    )
    work_offsets = torch.tensor(
        work_offsets_list, device=block_table.device, dtype=torch.int32
    )
    return work_tiles, work_offsets


def indexer_score_cutedsl_v3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    work_tiles: torch.Tensor,
    work_offsets: torch.Tensor,
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    num_sms: int = 148,
) -> None:
    _, num_heads, head_dim = q.shape

    Sm100MSAIndexerScoreKernelV3.compile(num_heads, head_dim)(
        q,
        k_cache,
        block_table,
        work_tiles,
        work_offsets,
        score,
        cu_seqlens_q,
        context_lens,
        num_sms,
    )
