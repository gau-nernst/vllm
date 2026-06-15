# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark MiniMax M3 indexer score kernel for prefill workloads."""

from __future__ import annotations

# ruff: noqa: E402, I001

import argparse
import os
import random
import statistics
from collections.abc import Callable

os.environ["CUTE_DSL_KEEP_PTX"] = "1"
os.environ["CUTE_DSL_KEEP_CUBIN"] = "1"
os.environ["CUTE_DSL_LINEINFO"] = "1"
os.environ["CUTE_DSL_DUMP_DIR"] = "./cutedsl_dump"
os.environ["CUTE_DSL_NO_CACHE"] = "1"
os.makedirs(os.environ["CUTE_DSL_DUMP_DIR"], exist_ok=True)

import pandas as pd
import torch
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.minimax_m3.common.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    _index_block_score_kernel,
)
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

TOTAL_INDEX_HEADS = 4
INDEX_HEAD_DIM = 128
BLOCK_SIZE_Q = 64
SM_SCALE = INDEX_HEAD_DIM**-0.5
DTYPE_BYTES = 2
SCORE_BYTES = 4
LOG2_E = 1.4426950409
MsaOps = tuple[Callable[..., object], Callable[..., object], Callable[..., object]]
CuteDslOp = Callable[
    [
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ],
    None,
]
CuteDslV3Ops = tuple[
    Callable[..., tuple[torch.Tensor, torch.Tensor]], Callable[..., None]
]


def _bench(fn: Callable[[], object]) -> float:
    torch.accelerator.synchronize()
    return statistics.median(bench_gpu_time_with_cupti(fn)) * 1e3


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _make_inputs(
    q_lens: list[int],
    context_lens: list[int],
    *,
    num_index_heads: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_lens = [q_len + context_len for q_len, context_len in zip(q_lens, context_lens)]
    total_q = sum(q_lens)
    max_blocks = _cdiv(max(seq_lens), SPARSE_BLOCK_SIZE)
    index_kv_cache = torch.randn(
        len(q_lens) * max_blocks,
        SPARSE_BLOCK_SIZE,
        INDEX_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    idx_q = torch.randn(
        total_q,
        num_index_heads,
        INDEX_HEAD_DIM,
        device=device,
        dtype=dtype,
    )
    pages = torch.arange(len(q_lens) * max_blocks, device=device, dtype=torch.int32)
    block_table = pages.reshape(len(q_lens), max_blocks)
    return idx_q, index_kv_cache, block_table


def _run_triton_prefill_score(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    score: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    context_lens: torch.Tensor,
    max_query_len: int,
) -> None:
    num_index_heads = idx_q.shape[1]
    grid = (
        triton.cdiv(max_query_len, BLOCK_SIZE_Q),
        seq_lens.numel() * num_index_heads,
    )
    _index_block_score_kernel[grid](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        context_lens,
        num_index_heads,
        INDEX_HEAD_DIM,
        SM_SCALE,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
    )


def _make_msa_case(
    q_lens: list[int],
    context_lens: list[int],
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    fmha_sm100_plan: Callable[..., object],
) -> tuple[torch.Tensor, object, torch.Tensor, torch.Tensor]:
    q_lens_cpu = torch.tensor(q_lens, dtype=torch.int32)
    seq_lens_cpu = torch.tensor(
        [q_len + context_len for q_len, context_len in zip(q_lens, context_lens)],
        dtype=torch.int32,
    )
    context_lens_cpu = torch.tensor(context_lens, dtype=torch.int32)
    plan = fmha_sm100_plan(
        q_lens_cpu,
        seq_lens_cpu,
        idx_q.shape[1],
        num_kv_heads=1,
        qo_offset=context_lens_cpu,
        page_size=SPARSE_BLOCK_SIZE,
        output_maxscore=True,
        causal=True,
        device=idx_q.device,
    )
    num_pages = [
        _cdiv(int(seq_len), SPARSE_BLOCK_SIZE) for seq_len in seq_lens_cpu.tolist()
    ]
    kv_indices = torch.empty(
        sum(num_pages), device=block_table.device, dtype=torch.int32
    )
    offset = 0
    for request_id, pages in enumerate(num_pages):
        kv_indices[offset : offset + pages].copy_(block_table[request_id, :pages])
        offset += pages
    max_seq_len = max(
        q_len + context_len for q_len, context_len in zip(q_lens, context_lens)
    )
    max_k_tiles = round_up(_cdiv(max_seq_len, SPARSE_BLOCK_SIZE), 128)
    max_score = torch.empty(
        (idx_q.shape[1], max_k_tiles, idx_q.shape[0]),
        dtype=torch.float32,
        device=idx_q.device,
    )
    return index_kv_cache[:, None], plan, kv_indices, max_score


def _msa_tuned_num_kv_splits(num_index_heads: int, max_context_len: int) -> int:
    if num_index_heads == 4 and max_context_len >= 65536:
        return 16
    if num_index_heads == 4 and max_context_len >= 8192:
        return 4
    return -1


def _cutedsl_v1_num_sms(
    q_lens: list[int],
    context_lens: list[int],
    num_index_heads: int,
) -> int:
    if len(q_lens) != 1:
        return 148

    num_q_tiles = _cdiv(q_lens[0], 128)
    num_q_head_tiles = num_q_tiles * num_index_heads
    context_len = context_lens[0]

    if num_index_heads == 4:
        if context_len == 0:
            return 2 * num_q_head_tiles
        if context_len < 65536:
            return 4 * num_q_head_tiles
        return 148

    if num_index_heads == 1:
        if context_len == 0:
            return 5 * num_q_head_tiles
        if context_len < 65536:
            return 4 * num_q_head_tiles
        return 148

    return 148


def _make_msa_tuned_case(
    q_lens: list[int],
    context_lens: list[int],
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    fmha_sm100_plan: Callable[..., object],
) -> tuple[torch.Tensor, object, torch.Tensor, torch.Tensor]:
    tuned_q_lens: list[int] = []
    tuned_seq_lens: list[int] = []
    tuned_q_offsets: list[int] = []
    tuned_request_ids: list[int] = []

    msa_tuned_chunk_size = 128
    for request_id, (q_len, context_len) in enumerate(zip(q_lens, context_lens)):
        for q_start in range(0, q_len, msa_tuned_chunk_size):
            chunk_len = min(msa_tuned_chunk_size, q_len - q_start)
            q_end = q_start + chunk_len
            tuned_q_lens.append(chunk_len)
            tuned_seq_lens.append(context_len + q_end)
            tuned_q_offsets.append(context_len + q_start)
            tuned_request_ids.append(request_id)

    q_lens_cpu = torch.tensor(tuned_q_lens, dtype=torch.int32)
    seq_lens_cpu = torch.tensor(tuned_seq_lens, dtype=torch.int32)
    q_offsets_cpu = torch.tensor(tuned_q_offsets, dtype=torch.int32)
    plan = fmha_sm100_plan(
        q_lens_cpu,
        seq_lens_cpu,
        idx_q.shape[1],
        num_kv_heads=1,
        qo_offset=q_offsets_cpu,
        num_kv_splits=_msa_tuned_num_kv_splits(idx_q.shape[1], max(context_lens)),
        page_size=SPARSE_BLOCK_SIZE,
        output_maxscore=True,
        causal=True,
        device=idx_q.device,
        split_prefill_decode=False,
    )
    num_pages = [_cdiv(seq_len, SPARSE_BLOCK_SIZE) for seq_len in tuned_seq_lens]
    kv_indices = torch.empty(
        sum(num_pages), device=block_table.device, dtype=torch.int32
    )
    offset = 0
    for request_id, pages in zip(tuned_request_ids, num_pages):
        kv_indices[offset : offset + pages].copy_(block_table[request_id, :pages])
        offset += pages
    max_seq_len = max(
        q_len + context_len for q_len, context_len in zip(q_lens, context_lens)
    )
    max_k_tiles = round_up(_cdiv(max_seq_len, SPARSE_BLOCK_SIZE), 128)
    max_score = torch.empty(
        (idx_q.shape[1], max_k_tiles, idx_q.shape[0]),
        dtype=torch.float32,
        device=idx_q.device,
    )
    return index_kv_cache[:, None], plan, kv_indices, max_score


def _sample_indices(length: int) -> list[int]:
    return sorted({0, length // 2, length - 1})


def _print_v3_work_schedule(
    q_lens: list[int],
    context_lens: list[int],
    work_tiles: torch.Tensor,
    work_offsets: torch.Tensor,
    num_index_heads: int,
) -> None:
    work_tiles_cpu = work_tiles.cpu().tolist()
    work_offsets_cpu = work_offsets.cpu().tolist()
    k_tiles_by_req_q: dict[tuple[int, int], int] = {}
    for req_id, (q_len, context_len) in enumerate(zip(q_lens, context_lens)):
        num_q_tiles = _cdiv(q_len, 128)
        for q_tile_id in range(num_q_tiles):
            q_tile_end = min((q_tile_id + 1) * 128, q_len)
            k_tiles_by_req_q[(req_id, q_tile_id)] = _cdiv(context_len + q_tile_end, 128)

    print()
    print("v3 work schedule")
    print(
        f"num_index_heads={num_index_heads} "
        f"grid_x={len(work_offsets_cpu) - 1} "
        f"num_work_tiles={len(work_tiles_cpu)}"
    )
    print("requests:")
    for req_id, (q_len, context_len) in enumerate(zip(q_lens, context_lens)):
        req_q_tiles = _cdiv(q_len, 128)
        req_qk_tiles = sum(
            k_tiles_by_req_q[(req_id, q_tile_id)] for q_tile_id in range(req_q_tiles)
        )
        print(
            f"  req={req_id} q_len={q_len} context={context_len} "
            f"q_tiles={req_q_tiles} qk_tiles={req_qk_tiles}"
        )
    print("ctas:")
    for cta_id, (begin, end) in enumerate(zip(work_offsets_cpu, work_offsets_cpu[1:])):
        entries = work_tiles_cpu[begin:end]
        qk_tiles = sum(group_len for _, _, _, group_len in entries)
        if not entries:
            print(f"  cta={cta_id:03d} qk_tiles=0 work=[]")
            continue
        parts = []
        for req_id, q_tile_id, k_start, group_len in entries:
            k_total = k_tiles_by_req_q[(req_id, q_tile_id)]
            parts.append(
                f"r{req_id}:q{q_tile_id}:k[{k_start},{k_start + group_len})/{k_total}"
            )
        print(f"  cta={cta_id:03d} qk_tiles={qk_tiles:3d} work={' '.join(parts)}")


def _check_scores(
    idx_q: torch.Tensor,
    index_kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    triton_score: torch.Tensor,
    msa_score: torch.Tensor | None,
    msa_tuned_score: torch.Tensor | None,
    cutedsl_score: torch.Tensor | None,
    cutedsl_v3_score: torch.Tensor | None,
    q_lens: list[int],
    context_lens: list[int],
) -> None:
    # return
    q_offset = 0
    for request_id, (q_len, context_len) in enumerate(zip(q_lens, context_lens)):
        for q_idx in _sample_indices(q_len):
            token_idx = q_offset + q_idx
            visible_blocks = _cdiv(context_len + q_idx + 1, SPARSE_BLOCK_SIZE)
            for block_idx in _sample_indices(visible_blocks):
                page = int(block_table[request_id, block_idx])
                kv_len = context_len + q_idx + 1
                valid_positions = min(
                    SPARSE_BLOCK_SIZE,
                    kv_len - block_idx * SPARSE_BLOCK_SIZE,
                )
                k = index_kv_cache[page, :valid_positions].float()
                q = idx_q[token_idx].float()
                reference = torch.matmul(q, k.T).amax(dim=1)
                torch.testing.assert_close(
                    triton_score[:, token_idx, block_idx],
                    reference * SM_SCALE * LOG2_E,
                    atol=1e-2,
                    rtol=1e-2,
                )
                if msa_score is not None:
                    torch.testing.assert_close(
                        msa_score[:, block_idx, token_idx],
                        reference,
                        atol=1e-2,
                        rtol=1e-2,
                    )
                if msa_tuned_score is not None:
                    torch.testing.assert_close(
                        msa_tuned_score[:, block_idx, token_idx],
                        reference,
                        atol=1e-2,
                        rtol=1e-2,
                    )
                if cutedsl_score is not None:
                    torch.testing.assert_close(
                        cutedsl_score[:, token_idx, block_idx],
                        reference,
                        atol=1e-2,
                        rtol=1e-2,
                    )
                if cutedsl_v3_score is not None:
                    torch.testing.assert_close(
                        cutedsl_v3_score[:, token_idx, block_idx],
                        reference,
                        atol=1e-2,
                        rtol=1e-2,
                    )
        q_offset += q_len


def _bench_case(
    q_lens: list[int],
    context_lens: list[int],
    *,
    msa_ops: MsaOps | None,
    cutedsl_op: CuteDslOp | None,
    cutedsl_v3_ops: CuteDslV3Ops | None,
    num_index_heads: int,
    print_v3_schedule: bool = False,
) -> tuple[
    float,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    idx_q, index_kv_cache, block_table = _make_inputs(
        q_lens,
        context_lens,
        num_index_heads=num_index_heads,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_lens_tensor = torch.tensor(q_lens, device=idx_q.device, dtype=torch.int32)
    cu_seqlens_q = torch.empty(len(q_lens) + 1, device=idx_q.device, dtype=torch.int32)
    cu_seqlens_q[0] = 0
    torch.cumsum(q_lens_tensor, dim=0, out=cu_seqlens_q[1:])
    seq_lens_list = [
        q_len + context_len for q_len, context_len in zip(q_lens, context_lens)
    ]
    seq_lens = torch.tensor(seq_lens_list, device=idx_q.device, dtype=torch.int32)
    context_lens_tensor = torch.tensor(
        context_lens, device=idx_q.device, dtype=torch.int32
    )
    max_query_len = max(q_lens)
    score_block_stride = round_up(_cdiv(max(seq_lens_list), SPARSE_BLOCK_SIZE), 16)
    triton_score = torch.empty(
        (idx_q.shape[1], idx_q.shape[0], score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    cutedsl_score = None
    if cutedsl_op is not None:
        cutedsl_score = torch.empty_like(triton_score)
    cutedsl_v3_score = None
    cutedsl_v3_work_tiles = None
    cutedsl_v3_work_offsets = None
    if cutedsl_v3_ops is not None:
        prepare_cutedsl_v3_work_tiles, _ = cutedsl_v3_ops
        cutedsl_v3_score = torch.empty_like(triton_score)
        cutedsl_v3_work_tiles, cutedsl_v3_work_offsets = prepare_cutedsl_v3_work_tiles(
            cu_seqlens_q,
            context_lens_tensor,
            block_table,
            idx_q.shape[1],
        )
        if print_v3_schedule:
            _print_v3_work_schedule(
                q_lens,
                context_lens,
                cutedsl_v3_work_tiles,
                cutedsl_v3_work_offsets,
                idx_q.shape[1],
            )

    def run_triton() -> None:
        _run_triton_prefill_score(
            idx_q,
            index_kv_cache,
            block_table,
            triton_score,
            cu_seqlens_q,
            seq_lens,
            context_lens_tensor,
            max_query_len,
        )

    triton_score.fill_(-float("inf"))
    run_triton()
    torch.accelerator.synchronize()

    def run_cutedsl() -> None:
        assert cutedsl_score is not None
        assert cutedsl_op is not None
        cutedsl_op(
            idx_q,
            index_kv_cache,
            block_table,
            cutedsl_score,
            cu_seqlens_q,
            context_lens_tensor,
            max_query_len,
            _cutedsl_v1_num_sms(q_lens, context_lens, idx_q.shape[1]),
        )

    if cutedsl_score is not None:
        cutedsl_score.fill_(-float("inf"))
        run_cutedsl()
        torch.accelerator.synchronize()

    def run_cutedsl_v3() -> None:
        assert cutedsl_v3_score is not None
        assert cutedsl_v3_work_tiles is not None
        assert cutedsl_v3_work_offsets is not None
        assert cutedsl_v3_ops is not None
        _, cutedsl_v3_op = cutedsl_v3_ops
        cutedsl_v3_op(
            idx_q,
            index_kv_cache,
            block_table,
            cutedsl_v3_work_tiles,
            cutedsl_v3_work_offsets,
            cutedsl_v3_score,
            cu_seqlens_q,
            context_lens_tensor,
        )

    if cutedsl_v3_score is not None:
        cutedsl_v3_score.fill_(-float("inf"))
        run_cutedsl_v3()
        torch.accelerator.synchronize()

    if msa_ops is None:
        _check_scores(
            idx_q,
            index_kv_cache,
            block_table,
            triton_score,
            None,
            None,
            cutedsl_score,
            cutedsl_v3_score,
            q_lens,
            context_lens,
        )
        cutedsl_us = _bench(run_cutedsl) if cutedsl_score is not None else None
        cutedsl_v3_us = _bench(run_cutedsl_v3) if cutedsl_v3_score is not None else None
        return (
            _bench(run_triton),
            None,
            None,
            cutedsl_us,
            cutedsl_v3_us,
        )

    fmha_sm100, fmha_sm100_plan, _fmha_sm100 = msa_ops
    msa_index_kv_cache, msa_plan, msa_kv_indices, msa_score = _make_msa_case(
        q_lens, context_lens, idx_q, index_kv_cache, block_table, fmha_sm100_plan
    )
    (
        msa_tuned_index_kv_cache,
        msa_tuned_plan,
        msa_tuned_kv_indices,
        msa_tuned_score,
    ) = _make_msa_tuned_case(
        q_lens, context_lens, idx_q, index_kv_cache, block_table, fmha_sm100_plan
    )

    def run_msa() -> None:
        fmha_sm100(
            idx_q,
            msa_index_kv_cache,
            msa_index_kv_cache,
            msa_plan,
            kv_indices=msa_kv_indices,
            max_score=msa_score,
            output_maxscore=True,
            output_o=False,
            check_input_valid=False,
            sm_scale=SM_SCALE,
        )

    msa_score.fill_(-float("inf"))
    run_msa()
    torch.accelerator.synchronize()

    def run_msa_tuned() -> None:
        _fmha_sm100(
            idx_q,
            msa_tuned_index_kv_cache,
            msa_tuned_index_kv_cache,
            msa_tuned_plan[3],
            kv_indices=msa_tuned_kv_indices,
            max_score=msa_tuned_score,
            output_maxscore=True,
            output_o=False,
            check_input_valid=False,
            sm_scale=SM_SCALE,
        )

    msa_tuned_score.fill_(-float("inf"))
    run_msa_tuned()
    torch.accelerator.synchronize()
    _check_scores(
        idx_q,
        index_kv_cache,
        block_table,
        triton_score,
        msa_score,
        msa_tuned_score,
        cutedsl_score,
        cutedsl_v3_score,
        q_lens,
        context_lens,
    )

    cutedsl_us = _bench(run_cutedsl) if cutedsl_score is not None else None
    cutedsl_v3_us = _bench(run_cutedsl_v3) if cutedsl_v3_score is not None else None
    return (
        _bench(run_triton),
        _bench(run_msa),
        _bench(run_msa_tuned),
        cutedsl_us,
        cutedsl_v3_us,
    )


def _prefill_score_flops(
    q_lens: list[int],
    context_lens: list[int],
    num_index_heads: int,
) -> int:
    total_qk_pairs = 0
    for q_len, context_len in zip(q_lens, context_lens):
        for q_idx in range(q_len):
            total_qk_pairs += context_len + q_idx + 1
    return num_index_heads * total_qk_pairs * 2 * INDEX_HEAD_DIM


def _tflops(flops: int, us: float) -> float:
    return flops / (us * 1e6)


def _print_table(
    rows: list[dict[str, float | int | str]],
    *,
    title: str,
    columns: list[str],
) -> None:
    df = pd.DataFrame(rows)
    print(title)
    print(df[columns].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def _add_result(
    rows: list[dict[str, float | int | str]],
    row: dict[str, float | int | str],
    q_lens: list[int],
    context_lens: list[int],
    *,
    msa_ops: MsaOps | None,
    cutedsl_op: CuteDslOp | None,
    cutedsl_v3_ops: CuteDslV3Ops | None,
    num_index_heads: int,
    print_v3_schedule: bool = False,
) -> None:
    (
        triton_us,
        msa_us,
        msa_tuned_us,
        cutedsl_us,
        cutedsl_v3_us,
    ) = _bench_case(
        q_lens,
        context_lens,
        msa_ops=msa_ops,
        cutedsl_op=cutedsl_op,
        cutedsl_v3_ops=cutedsl_v3_ops,
        num_index_heads=num_index_heads,
        print_v3_schedule=print_v3_schedule,
    )
    flops = _prefill_score_flops(q_lens, context_lens, num_index_heads)
    row["triton_us"] = triton_us
    row["triton_tflops"] = _tflops(flops, triton_us)
    if msa_us is not None:
        row["msa_us"] = msa_us
        row["msa_tflops"] = _tflops(flops, msa_us)
    if msa_tuned_us is not None:
        row["msa_tuned_us"] = msa_tuned_us
        row["msa_tuned_tflops"] = _tflops(flops, msa_tuned_us)
    if cutedsl_us is not None:
        row["cutedsl_us"] = cutedsl_us
        row["cutedsl_tflops"] = _tflops(flops, cutedsl_us)
    if cutedsl_v3_us is not None:
        row["cutedsl_v3_us"] = cutedsl_v3_us
        row["cutedsl_v3_tflops"] = _tflops(flops, cutedsl_v3_us)
    rows.append(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MiniMax M3 BF16 prefill indexer score kernel."
    )
    parser.add_argument("--total-tokens", type=int, default=8192)
    parser.add_argument(
        "--context-lens",
        type=int,
        nargs="*",
        default=[0, 8192, 16384, 32768, 65536, 131072],
        help="Context lengths for the single-request workload.",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16],
        help="Request counts for the mixed-request workload.",
    )
    parser.add_argument(
        "--max-random-context-len",
        type=int,
        default=131072,
        help="Maximum context length for deterministic mixed workloads.",
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--csv", action="store_true")
    parser.add_argument("--print-v3-schedule", action="store_true")
    args = parser.parse_args(argv)

    if not torch.accelerator.is_available():
        raise RuntimeError("CUDA is required for MiniMax M3 indexer kernels.")

    if args.tp_size < 1:
        raise ValueError("--tp-size must be positive.")
    if args.tp_size <= TOTAL_INDEX_HEADS:
        if TOTAL_INDEX_HEADS % args.tp_size != 0:
            raise ValueError(
                f"tp_size={args.tp_size} must divide "
                f"TOTAL_INDEX_HEADS={TOTAL_INDEX_HEADS}."
            )
        num_index_heads = TOTAL_INDEX_HEADS // args.tp_size
    else:
        if args.tp_size % TOTAL_INDEX_HEADS != 0:
            raise ValueError(
                f"tp_size={args.tp_size} must divide "
                f"TOTAL_INDEX_HEADS={TOTAL_INDEX_HEADS}."
            )
        num_index_heads = 1

    try:
        from fmha_sm100 import fmha_sm100, fmha_sm100_plan
        from fmha_sm100.api import _fmha_sm100
    except ModuleNotFoundError as exc:
        if exc.name != "fmha_sm100":
            raise
        msa_ops = None
    else:
        msa_ops = fmha_sm100, fmha_sm100_plan, _fmha_sm100
    try:
        from benchmark_m3.indexer_score_cutedsl import indexer_score_cutedsl
    except ModuleNotFoundError as exc:
        if exc.name == "benchmark_m3":
            try:
                from indexer_score_cutedsl import indexer_score_cutedsl
            except ModuleNotFoundError:
                cutedsl_op = None
            else:
                cutedsl_op = indexer_score_cutedsl
        else:
            cutedsl_op = None
    else:
        cutedsl_op = indexer_score_cutedsl
    try:
        from benchmark_m3.indexer_score_cutedsl_v3 import (
            indexer_score_cutedsl_v3,
            prepare_indexer_score_cutedsl_v3_work_tiles,
        )
    except ModuleNotFoundError as exc:
        if exc.name == "benchmark_m3":
            try:
                from indexer_score_cutedsl_v3 import (
                    indexer_score_cutedsl_v3,
                    prepare_indexer_score_cutedsl_v3_work_tiles,
                )
            except ModuleNotFoundError:
                cutedsl_v3_ops = None
            else:
                cutedsl_v3_ops = (
                    prepare_indexer_score_cutedsl_v3_work_tiles,
                    indexer_score_cutedsl_v3,
                )
        else:
            cutedsl_v3_ops = None
    else:
        cutedsl_v3_ops = (
            prepare_indexer_score_cutedsl_v3_work_tiles,
            indexer_score_cutedsl_v3,
        )
    rng = random.Random()

    single_rows: list[dict[str, float | int | str]] = []
    mixed_rows: list[dict[str, float | int | str]] = []

    for context_len in args.context_lens:
        q_lens = [args.total_tokens]
        context_lens = [context_len]
        _add_result(
            single_rows,
            {
                "workload": "single",
                "num_requests": 1,
                "total_tokens": args.total_tokens,
                "context": context_len,
            },
            q_lens,
            context_lens,
            msa_ops=msa_ops,
            cutedsl_op=cutedsl_op,
            cutedsl_v3_ops=cutedsl_v3_ops,
            num_index_heads=num_index_heads,
        )

    for num_requests in args.num_requests:
        if num_requests < 1:
            raise ValueError("num_requests must be positive.")
        if num_requests > args.total_tokens:
            raise ValueError("num_requests cannot exceed total_tokens.")
        cuts = sorted(rng.sample(range(1, args.total_tokens), num_requests - 1))
        q_lens = [
            end - start for start, end in zip([0] + cuts, cuts + [args.total_tokens])
        ]
        max_context_units = args.max_random_context_len // SPARSE_BLOCK_SIZE
        context_lens = [
            rng.randint(0, max_context_units) * SPARSE_BLOCK_SIZE
            for _ in range(num_requests)
        ]
        _add_result(
            mixed_rows,
            {
                "workload": "mixed",
                "num_requests": num_requests,
                "total_tokens": args.total_tokens,
                "q_range": f"[{min(q_lens)},{max(q_lens)}]",
                "context_range": f"[{min(context_lens)},{max(context_lens)}]",
            },
            q_lens,
            context_lens,
            msa_ops=msa_ops,
            cutedsl_op=cutedsl_op,
            cutedsl_v3_ops=cutedsl_v3_ops,
            num_index_heads=num_index_heads,
            print_v3_schedule=args.print_v3_schedule,
        )

    if args.csv:
        print(pd.DataFrame(single_rows + mixed_rows).to_csv(index=False), end="")
        return

    print("kernel: Triton prefill score")
    print(f"tp_size={args.tp_size} num_index_heads={num_index_heads}")
    comparison = "MSA dense FMHA max-score"
    if msa_ops is None:
        comparison += " (skipped: not importable)"
    print(f"comparison: {comparison}")
    msa_tuned_comparison = "MSA tuned segmented max-score"
    if msa_ops is None:
        msa_tuned_comparison += " (skipped: not importable)"
    print(f"comparison: {msa_tuned_comparison}")
    cutedsl_comparison = "CuTe DSL indexer score"
    if cutedsl_op is None:
        cutedsl_comparison += " (skipped: not importable)"
    print(f"comparison: {cutedsl_comparison}")
    cutedsl_v3_comparison = "CuTe DSL v3 grouped-K indexer score"
    if cutedsl_v3_ops is None:
        cutedsl_v3_comparison += " (skipped: not importable)"
    print(f"comparison: {cutedsl_v3_comparison}")
    base_columns = [
        "num_requests",
        "total_tokens",
        "context",
        "q_range",
        "context_range",
        "triton_us",
    ]
    if msa_ops is not None:
        base_columns.append("msa_us")
        base_columns.append("msa_tuned_us")
    if cutedsl_op is not None:
        base_columns.append("cutedsl_us")
    if cutedsl_v3_ops is not None:
        base_columns.append("cutedsl_v3_us")
    base_columns.extend(["triton_tflops"])
    if msa_ops is not None:
        base_columns.append("msa_tflops")
        base_columns.append("msa_tuned_tflops")
    if cutedsl_op is not None:
        base_columns.append("cutedsl_tflops")
    if cutedsl_v3_ops is not None:
        base_columns.append("cutedsl_v3_tflops")
    columns = base_columns
    print()
    _print_table(
        single_rows,
        title="workload: single",
        columns=[col for col in columns if col in single_rows[0]],
    )
    print()
    _print_table(
        mixed_rows,
        title="workload: mixed",
        columns=[col for col in columns if col in mixed_rows[0]],
    )


if __name__ == "__main__":
    main()
