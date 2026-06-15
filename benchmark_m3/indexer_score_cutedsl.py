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


class Sm100MSAIndexerScoreKernel:
    BLOCK_Q = 128
    BLOCK_K = 128
    num_stages = 4

    def __init__(self, num_heads: int, head_dim: int = 128):
        self.num_heads = num_heads
        self.head_dim = head_dim

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,  # [total_tokens, num_heads, head_dim]
        gK_cache: cute.Tensor,  # [num_pages, page_size, head_dim]
        block_table: cute.Tensor,  # [num_seqs, max_pages]
        score: cute.Tensor,  # [num_heads, total_tokens, max_pages]
        cu_seqlens_q: cute.Tensor,  # [num_seqs+1]
        context_lens: cute.Tensor,  # [num_seqs]
        max_query_len: Int32,
        num_sms: Int32,
        stream: CUstream,
    ):
        BLOCK_Q = self.BLOCK_Q
        BLOCK_K = self.BLOCK_K
        num_stages = self.num_stages
        head_dim = self.head_dim
        num_heads = self.num_heads

        num_seqs = block_table.shape[0]
        num_q_tiles = cute.ceil_div(max_query_len, BLOCK_Q)
        num_q_head_tiles = num_q_tiles * num_heads
        split_k = max(1, num_sms // num_q_head_tiles)
        grid = (num_q_head_tiles, split_k, num_seqs)
        block = (32 * 6, 1, 1)

        tma_g2s = cpasync.CopyBulkTensorTileG2SOp()
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        sQ_layout = cute.make_layout(
            (BLOCK_Q, 1, (64, self.head_dim // 64)),
            stride=(64, 0, (1, BLOCK_Q * 64)),
        )
        sQ_layout = cute.make_composed_layout(swizzle_128B, 0, sQ_layout)
        Q_tma = cpasync.make_tiled_tma_atom(
            tma_g2s,
            cute.logical_divide(gQ, (None, None, 64)),
            sQ_layout,
            cta_tiler=(BLOCK_Q, 1, head_dim),
        )

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
            score,
            cu_seqlens_q,
            context_lens,
            split_k,
        ).launch(grid=grid, block=block, stream=stream)

    @cute.kernel
    def kernel(
        self,
        Q_tma: cpasync.TmaInfo,
        K_tma: cpasync.TmaInfo,
        block_table: cute.Tensor,  # [num_seqs, max_pages]
        score: cute.Tensor,  # [num_heads, total_tokens, max_pages]
        cu_seqlens_q: cute.Tensor,  # [num_seqs+1]
        context_lens: cute.Tensor,  # [num_seqs]
        split_k: Int32,
    ):
        tid, _, _ = cute.arch.thread_idx()
        bid_q_head, split_id, seq_id = cute.arch.block_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)

        BLOCK_Q = self.BLOCK_Q
        BLOCK_K = self.BLOCK_K
        num_stages = self.num_stages
        head_dim = self.head_dim
        num_heads = self.num_heads
        bid_q = bid_q_head // num_heads
        head_id = bid_q_head - bid_q * num_heads

        bos = cu_seqlens_q[seq_id]
        eos = cu_seqlens_q[seq_id + 1]
        q_len = eos - bos

        context_len = context_lens[seq_id]
        max_seqlen = context_len + min((bid_q + 1) * BLOCK_Q, q_len)
        num_k_blocks = cute.ceil_div(max_seqlen, BLOCK_K)

        def allocate_tensor(smem, dtype, layout):
            return smem.allocate_tensor(
                dtype, layout.outer, byte_alignment=128, swizzle=layout.inner
            )

        if bid_q * BLOCK_Q < q_len and split_id < num_k_blocks:
            smem = cutlass.utils.SmemAllocator()
            sQ = allocate_tensor(smem, BFloat16, Q_tma.smem_layout)
            sK = allocate_tensor(smem, BFloat16, K_tma.smem_layout)[0, None, None, None]

            tma_mbar = smem.allocate_array(Int64, num_stages)
            mma_mbar = smem.allocate_array(Int64, num_stages)
            epi_mbar = smem.allocate_array(Int64, num_stages)

            taddr = smem.allocate_array(Int32, 1)
            assert BLOCK_K * num_stages <= 512

            if warp_id == 0:
                with cute.arch.elect_one():
                    for i in cutlass.range_constexpr(num_stages):
                        cute.arch.mbarrier_init(tma_mbar + i, 1)
                        cute.arch.mbarrier_init(mma_mbar + i, 1)
                        cute.arch.mbarrier_init(epi_mbar + i, 128)
                    cute.arch.mbarrier_init_fence()
            elif warp_id == 1:
                cpasync.prefetch_descriptor(Q_tma.atom)
                cpasync.prefetch_descriptor(K_tma.atom)
            cute.arch.sync_threads()

            if warp_id == 5:
                stage_id = 0
                parity = 1

                gQ_tile = cute.local_tile(
                    cute.domain_offset(
                        (bos, 0),
                        Q_tma.tma_tensor[None, head_id, None],
                    ),
                    tiler=(BLOCK_Q, head_dim),
                    coord=(bid_q, 0),
                )
                page_id0 = block_table[seq_id, split_id]
                gK_tile0 = K_tma.tma_tensor[page_id0, None, None]

                with cute.arch.elect_one():
                    QK_size = (BLOCK_Q + BLOCK_K) * head_dim * 2
                    cute.arch.mbarrier_arrive_and_expect_tx(tma_mbar, QK_size)
                simple_tma_copy(Q_tma.atom, gQ_tile, sQ, tma_mbar)
                simple_tma_copy(
                    K_tma.atom, gK_tile0, sK[None, None, stage_id], tma_mbar
                )

                stage_id = (stage_id + 1) % num_stages
                if stage_id == 0:
                    parity ^= 1

                for block_id in range(split_id + split_k, num_k_blocks, split_k):
                    page_id = block_table[seq_id, block_id]
                    gK_tile = K_tma.tma_tensor[page_id, None, None]
                    mbar = tma_mbar + stage_id

                    cute.arch.mbarrier_wait(mma_mbar + stage_id, parity)

                    with cute.arch.elect_one():
                        K_size = BLOCK_K * head_dim * 2
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar, K_size)
                    simple_tma_copy(K_tma.atom, gK_tile, sK[None, None, stage_id], mbar)

                    stage_id = (stage_id + 1) % num_stages
                    if stage_id == 0:
                        parity ^= 1

            elif warp_id == 4:
                _tcgen05.alloc(taddr)

                stage_id = 0
                parity = 0

                qdesc_template = _tcgen05.make_sdesc_128B_swizzle(BLOCK_Q * 128)
                kdesc_template = _tcgen05.make_sdesc_128B_swizzle(BLOCK_K * 128)
                idesc = _tcgen05.make_bf16_idesc(BLOCK_Q, BLOCK_K)

                qdesc_base = qdesc_template | (sQ.iterator.toint() >> 4)

                for _ in range(split_id, num_k_blocks, split_k):
                    k_addr = sK[None, None, stage_id].iterator.toint()
                    kdesc_base = kdesc_template | (k_addr >> 4)
                    acc_tmem = BLOCK_K * stage_id

                    cute.arch.mbarrier_wait(epi_mbar + stage_id, parity ^ 1)
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

            else:
                stage_id = 0
                parity = 0

                WIDTH = 64

                q_pos = bid_q * BLOCK_Q + tid
                first_q_pos = bid_q * BLOCK_Q
                num_full_blocks = min(
                    num_k_blocks,
                    (context_len + first_q_pos + 1) // BLOCK_K,
                )

                for block_id in range(split_id, num_k_blocks, split_k):
                    if warp_id == 0:
                        cute.arch.mbarrier_wait(mma_mbar + stage_id, parity)
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)
                    _tcgen05.fence_after_thread_sync()

                    max_score = cute.make_rmem_tensor(4, Float32)
                    max_score.fill(float("-inf"))

                    if block_id < num_full_blocks:
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

                        if q_pos < q_len:
                            score[head_id, bos + q_pos, block_id] = max_score[0]
                    else:
                        local_k_len = min(
                            context_len + q_pos + 1 - block_id * BLOCK_K,
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
                            max_score[0] = cute.arch.fmax(max_score[0], max_score[1])
                            max_score[2] = cute.arch.fmax(max_score[2], max_score[3])
                            max_score[0] = cute.arch.fmax(max_score[0], max_score[2])
                            score[head_id, bos + q_pos, block_id] = max_score[0]

                    stage_id = (stage_id + 1) % num_stages
                    if stage_id == 0:
                        parity ^= 1

                cute.arch.barrier(barrier_id=1, number_of_threads=128)
                _tcgen05.dealloc()

    @cache
    @staticmethod
    def compile(num_heads: int, head_dim: int = 128):
        total_tokens = cute.sym_int()
        num_seqs = cute.sym_int()
        BLOCK_K = Sm100MSAIndexerScoreKernel.BLOCK_K

        q = make_fake_tensor(
            BFloat16, (total_tokens, num_heads, head_dim), divisibility=16
        )
        k_cache = make_fake_tensor(
            BFloat16, (cute.sym_int(), BLOCK_K, head_dim), divisibility=16
        )
        block_table = make_fake_tensor(
            Int32, (num_seqs, cute.sym_int()), divisibility=1
        )
        score = make_fake_tensor(
            Float32, (num_heads, total_tokens, cute.sym_int()), divisibility=4
        )
        cu_seqlens_q = make_fake_tensor(Int32, (cute.sym_int(),), divisibility=1)
        context_lens = make_fake_tensor(Int32, (num_seqs,), divisibility=1)

        kernel = Sm100MSAIndexerScoreKernel(num_heads, head_dim)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            q,
            k_cache,
            block_table,
            score,
            cu_seqlens_q,
            context_lens,
            Int32(128),
            Int32(148),
            stream,
            options="--enable-tvm-ffi",
        )


def indexer_score_cutedsl(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    context_lens: torch.Tensor,
    max_query_len: int,
    num_sms: int = 148,
) -> None:
    _, num_heads, head_dim = q.shape

    Sm100MSAIndexerScoreKernel.compile(num_heads, head_dim)(
        q,
        k_cache,
        block_table,
        score,
        cu_seqlens_q,
        context_lens,
        max_query_len,
        num_sms,
    )
