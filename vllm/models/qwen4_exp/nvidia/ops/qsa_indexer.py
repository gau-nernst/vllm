# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA index selection."""

from typing import NamedTuple

import torch

import vllm.envs as envs
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    TritonWarmupTensor,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.triton_utils.allocation import set_triton_allocator

_TOPK_WORKSPACE_BYTES = 1024 * 1024
_DECODE_BLOCK_N = 64


@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    DECODE_QUERY_LEN_PADDED: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_start = tl.program_id(1) * TILES_PER_PROG
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tile_start + TILES_PER_PROG, tl.cdiv(max_visible, BLOCK_N))
    tile_end = tl.minimum(tile_end, tl.cdiv(NUM_COLUMNS, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit
def _gather_k_kernel(
    k_cache_ptr,
    page_table_ptr,
    k_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_k_req,
    max_cols,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CP: tl.constexpr,
) -> None:
    """Copy each request's useful compressed K columns into packed kc."""
    request = tl.program_id(0)
    cols = tl.program_id(1) * CP + tl.arange(0, CP)
    query_base = tl.load(query_start_loc_ptr)
    # useful columns = visible blocks at the request's last query row
    req_end = tl.load(query_start_loc_ptr + request + 1)
    useful = tl.load(visible_blocks_ptr + req_end - query_base - 1)
    live = cols < tl.minimum(max_cols, useful)
    logical_page = cols // PAGE_SIZE
    page_offset = cols % PAGE_SIZE
    phys = tl.load(
        page_table_ptr + request * stride_table_req + logical_page, mask=live, other=0
    )
    dims = tl.arange(0, HEAD_DIM)
    vals = tl.load(
        k_cache_ptr
        + phys[:, None].to(tl.int64) * stride_cache_block
        + page_offset[:, None] * stride_cache_token
        + dims[None, :],
        mask=live[:, None],
        other=0.0,
    )
    tl.store(
        k_ptr
        + request.to(tl.int64) * stride_k_req
        + cols[:, None].to(tl.int64) * HEAD_DIM
        + dims[None, :],
        vals,
        mask=live[:, None],
    )


@triton.jit(
    do_not_specialize=["num_rows", "query_offset", "num_k_cols", "num_logits_cols"]
)
def _qsa_mqa_prefill_kernel(
    q_ptr,
    k_ptr,
    k_desc,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_req,
    stride_logits_row,
    num_rows,
    query_offset,
    num_k_cols,
    num_logits_cols,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    USE_TMA: tl.constexpr,
    WS: tl.constexpr,
) -> None:
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base, query_offset
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base, query_end
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(visible_blocks_ptr + absolute_rows, mask=valid_rows, other=0)
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_k_cols, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED

    q_ptrs = (
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None]
    )
    q_mask = (
        (heads[None, :] < NUM_HEADS)
        & (q_rows[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM)
    )
    query = tl.load(q_ptrs, q_mask, other=0.0)

    column_offsets = tl.arange(0, BLOCK_N)
    k_row0 = request * (stride_k_req // HEAD_DIM)
    if USE_TMA:
        # Row bound = this request's end within the chunk, so TMA box
        # clipping handles request/chunk-boundary straddles (no store mask).
        store_desc = tl.make_tensor_descriptor(
            logits_ptr,
            [request_end - query_offset, num_logits_cols],
            [stride_logits_row, 1],
            [TILE_R, BLOCK_N],
        )
    for tile in tl.range(k_tile_start, k_tile_end, warp_specialize=WS):
        columns = tile * BLOCK_N + column_offsets
        if USE_TMA:
            keys = k_desc.load([k_row0 + tile * BLOCK_N, 0])
        else:
            k_ptrs = (
                k_ptr
                + (k_row0 + columns[:, None]).to(tl.int64) * HEAD_DIM
                + dims[None, :]
            )
            mask = (columns[:, None] < max_visible) & (dims[None, :] < HEAD_DIM)
            keys = tl.load(k_ptrs, mask, other=0.0, eviction_policy="evict_first")
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2)
        if USE_TMA:
            store_desc.store(
                [absolute_row_start - query_offset, tile * BLOCK_N], tl.trans(score)
            )
        else:
            logits_ptrs = (
                logits_ptr + rows[None, :] * stride_logits_row + columns[:, None]
            )
            store_mask = (
                valid_rows[None, :]
                & (columns[:, None] < visible[None, :])
                & (columns[:, None] < num_k_cols)
            )
            tl.store(logits_ptrs, score, store_mask)


@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    visible_blocks_ptr,
    output_ptr,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
) -> None:
    OUTPUT_WIDTH: tl.constexpr = BLOCK_TOPK * COMPRESS_RATIO + COMPRESS_RATIO - 1
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    visible_blocks = tl.load(visible_blocks_ptr + row)
    complete_blocks = tl.minimum(visible_blocks, BLOCK_TOPK)
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (columns < OUTPUT_WIDTH) & (is_expanded | is_tail) & (token >= 0)
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=columns < OUTPUT_WIDTH,
    )


def _decode_tiles_per_program(num_requests: int, columns: int) -> int:
    programs = num_requests * triton.cdiv(columns, _DECODE_BLOCK_N)
    if programs < 16384:
        return 1
    if programs < 32768:
        return 2
    if programs < 131072:
        return 4
    return 8


def _qsa_decode_warmup_profiles(
    max_dql: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
    columns: int,
) -> tuple[tuple[int, int], ...]:
    profiles: list[tuple[int, int]] = []
    for dql in range(1, max_dql + 1):
        max_reqs = min(max_num_reqs, max_num_batched_tokens // dql)
        requests_by_grouping: dict[int, int] = {}
        for num_requests in range(1, max_reqs + 1):
            tiles_per_program = _decode_tiles_per_program(num_requests, columns)
            requests_by_grouping.setdefault(tiles_per_program, num_requests)
        profiles.extend(
            (dql, num_requests) for num_requests in requests_by_grouping.values()
        )
    return tuple(profiles)


def warmup_qsa_mqa_paged_decode(
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    max_decode_query_len: int,
    max_num_reqs: int,
    max_num_batched_tokens: int,
) -> tuple[tuple[int, int], ...]:
    """Compile every reachable decode specialization without launching it."""

    page_size = k_cache.shape[1]
    page_table_width = page_table.shape[1]
    columns = page_table_width * page_size
    profiles = _qsa_decode_warmup_profiles(
        max_decode_query_len,
        max_num_reqs,
        max_num_batched_tokens,
        columns,
    )
    if not profiles:
        return ()

    k_cache_ptr = TritonWarmupTensor(k_cache.dtype, shape=tuple(k_cache.shape))
    page_table_ptr = TritonWarmupTensor(
        page_table.dtype,
        shape=(max_num_reqs, page_table_width),
    )
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)

    for decode_query_len, num_requests in profiles:
        num_rows = decode_query_len * num_requests
        q_ptr = TritonWarmupTensor(
            torch.bfloat16,
            shape=(num_rows, num_heads, head_dim),
        )
        logits_ptr = TritonWarmupTensor(
            torch.float32,
            shape=(num_rows, columns),
        )
        tiles_per_program = _decode_tiles_per_program(num_requests, columns)
        _qsa_mqa_paged_uniform_kernel.warmup(
            q_ptr,
            k_cache_ptr,
            page_table_ptr,
            visible_blocks_ptr,
            logits_ptr,
            num_heads * head_dim,
            head_dim,
            k_cache.stride(0),
            k_cache.stride(1),
            page_table.stride(0),
            columns,
            PAGE_SIZE=page_size,
            PAGE_TABLE_WIDTH=page_table_width,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            DECODE_QUERY_LEN=decode_query_len,
            BLOCK_N=_DECODE_BLOCK_N,
            TILES_PER_PROG=tiles_per_program,
            STAGES=2,
            num_warps=2,
            grid=(
                num_requests,
                triton.cdiv(columns, _DECODE_BLOCK_N * tiles_per_program),
            ),
        )
    return profiles


def warmup_qsa_mqa_prefill(
    k_cache: torch.Tensor, page_table: torch.Tensor, *, num_heads: int, head_dim: int
) -> tuple[str, ...]:
    """Compile every gathered-K prefill specialization without launching."""
    page_size = k_cache.shape[1]
    columns = 256  # >= TMA box dims; values are do_not_specialize
    num_reqs, rows = 1, 64
    q_ptr = TritonWarmupTensor(torch.bfloat16, shape=(rows, num_heads, head_dim))
    kc_ptr = TritonWarmupTensor(k_cache.dtype, shape=(num_reqs, columns, head_dim))
    logits_ptr = TritonWarmupTensor(torch.float32, shape=(rows, columns))
    visible_blocks_ptr = TritonWarmupTensor(torch.int32)
    query_start_loc_ptr = TritonWarmupTensor(torch.int32)

    _gather_k_kernel.warmup(
        k_cache,
        page_table,
        kc_ptr,
        query_start_loc_ptr,
        visible_blocks_ptr,
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        columns * head_dim,
        columns,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        CP=64,
        num_warps=8,
        grid=(num_reqs, 1),
    )

    warmed = []
    seen = set()
    for num_reqs_probe in (1, 8):
        config, use_tma = _prefill_plan(num_reqs_probe)
        if (config, use_tma) in seen:
            continue
        seen.add((config, use_tma))
        k_desc = None
        if use_tma:
            # Triton warmup needs a real (tiny) descriptor for the K-load
            # arg, and its box shape must match the config's BLOCK_N
            k_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                torch.empty(
                    columns, head_dim, dtype=k_cache.dtype, device=k_cache.device
                ),
                [config.BLOCK_N, head_dim],
            )
        _qsa_mqa_prefill_kernel.warmup(
            q_ptr,
            kc_ptr,
            k_desc,
            query_start_loc_ptr,
            visible_blocks_ptr,
            logits_ptr,
            num_heads * head_dim,
            head_dim,
            columns * head_dim,
            columns,
            rows,
            0,
            columns,
            columns,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            USE_TMA=use_tma,
            WS=use_tma,
            grid=(num_reqs, 1, 1),
            **config._asdict(),
        )
        warmed.append(config.BLOCK_N)
    return ("gather", *[f"block_n={bn}" for bn in warmed])


class _PrefillProfile(NamedTuple):
    TILE_R: int
    BLOCK_N: int
    K_TILES: int
    num_stages: int
    num_warps: int


def _prefill_plan(num_requests: int) -> tuple[_PrefillProfile, bool]:
    use_tma = current_platform.has_device_capability(90)

    # profile for pre-Hopper. 69.6 KB smem. untuned.
    if not use_tma:
        return _PrefillProfile(32, 64, 16, 2, 4), use_tma

    # profile for consumer Blackwell. untuned.
    if current_platform.is_device_capability_family(120):
        return _PrefillProfile(32, 64, 16, 2, 4), use_tma

    # profile for Hopper and Blackwell. tuned on GB300
    if num_requests >= 8:
        return _PrefillProfile(64, 128, 32, 2, 8), use_tma
    return _PrefillProfile(64, 128, 64, 3, 8), use_tma


def _prefill_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    max_query_len: int,
    logits_width: int,
    query_offset: int,
    num_queries: int,
    config: _PrefillProfile,
    use_tma: bool,
) -> torch.Tensor:
    assert query_start_loc.shape == (k.shape[0] + 1,)
    assert 0 <= query_offset <= query_offset + num_queries <= q.shape[0]
    assert 0 < logits_width <= k.shape[1]

    k_desc = None
    if use_tma:
        head_dim = k.shape[-1]
        k_desc = triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
            k.view(-1, head_dim), [config.BLOCK_N, head_dim]
        )
    logits = torch.empty(
        (num_queries, logits_width), dtype=torch.float32, device=q.device
    )
    grid = (
        k.shape[0],
        triton.cdiv(min(num_queries, max_query_len), config.TILE_R),
        triton.cdiv(logits_width, config.BLOCK_N * config.K_TILES),
    )
    _qsa_mqa_prefill_kernel[grid](
        q,
        k,
        k_desc,
        query_start_loc,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        k.stride(0),
        *logits.stride()[:-1],
        num_queries,
        query_offset,
        k.shape[1],
        logits_width,
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        USE_TMA=use_tma,
        WS=use_tma,
        **config._asdict(),
    )
    return logits


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    visible_blocks: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor,
) -> None:
    """Expand compressed blocks and compact the causal tail of the open group."""

    assert token_topk % compress_ratio == 0
    block_topk = token_topk // compress_ratio
    output_width = token_topk + compress_ratio - 1
    assert block_indices.shape == (query_positions.numel(), block_topk)
    assert visible_blocks.shape == query_positions.shape
    assert out.shape == (block_indices.shape[0], output_width)
    column_block = 256
    grid = (block_indices.shape[0], triton.cdiv(output_width, column_block))
    _expand_qsa_indices_kernel[grid](
        block_indices,
        query_positions,
        visible_blocks,
        out,
        *block_indices.stride(),
        *out.stride(),
        BLOCK_TOPK=block_topk,
        COMPRESS_RATIO=compress_ratio,
        COLUMN_BLOCK=column_block,
        num_warps=4,
    )


def _topk(
    logits: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    block_indices: torch.Tensor,
    topk_workspace: torch.Tensor,
) -> None:
    # similar dispatch logic as DeepSeek indexer
    block_topk = token_topk // compress_ratio
    use_cooperative_topk = (
        logits.shape[0] <= 64
        and logits.stride(0) % 4 == 0
        and current_platform.has_device_capability(90)
        and not current_platform.is_device_capability_family(120)
    )
    topk_op = (
        torch.ops._C.cooperative_topk
        if use_cooperative_topk
        else torch.ops._C.persistent_topk
    )
    topk_op(
        logits,
        visible_blocks,
        block_indices,
        topk_workspace,
        block_topk,
        logits.shape[1],
    )


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """Score and select compressed blocks for a request-major decode batch.

    Args:
        q: Query tensor shaped ``[num_requests * decode_query_len, heads,
            head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Request block table shaped ``[num_requests, max_pages]``.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        decode_query_len: Number of query tokens per request.
        block_indices: Compressed-index output buffer.
    """

    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    assert decode_query_len > 0 and q.shape[0] % decode_query_len == 0
    num_requests = q.shape[0] // decode_query_len
    assert page_table.shape[0] == num_requests
    assert visible_blocks.shape == (q.shape[0],)

    columns = page_table.shape[1] * k_cache.shape[1]
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    tiles_per_program = _decode_tiles_per_program(num_requests, columns)
    grid = (
        num_requests,
        triton.cdiv(columns, _DECODE_BLOCK_N * tiles_per_program),
    )
    _qsa_mqa_paged_uniform_kernel[grid](
        q,
        k_cache,
        page_table,
        visible_blocks,
        logits,
        *q.stride()[:-1],
        *k_cache.stride()[:2],
        *page_table.stride()[:-1],
        *logits.stride()[:-1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=page_table.shape[1],
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        DECODE_QUERY_LEN=decode_query_len,
        BLOCK_N=_DECODE_BLOCK_N,
        TILES_PER_PROG=tiles_per_program,
        STAGES=2,
        num_warps=2,
    )
    _topk(
        logits,
        visible_blocks,
        token_topk,
        compress_ratio,
        block_indices,
        torch.empty((_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device),
    )


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Score and select compressed prefill blocks in bounded chunks.

    Args:
        q: Packed prefill query tensor shaped ``[num_tokens, heads, head_dim]``.
        k_cache: Compressed key cache shaped ``[blocks, page_size, 1,
            head_dim]``.
        page_table: Block table shaped ``[num_requests, max_pages]``.
        query_start_loc: Packed prefill query offsets with a terminal offset.
            Offsets may share the base of a larger backing tensor.
        visible_blocks: Number of visible compressed blocks per query.
        token_topk: Number of logical tokens selected per query.
        compress_ratio: Number of logical tokens represented by a cache row.
        max_query_len: Maximum number of query tokens in one request.
        block_indices: Compressed-index output buffer.
        max_seq_len: Longest context length in the batch this step.
    """

    assert token_topk % compress_ratio == 0
    assert block_indices.shape == (q.shape[0], token_topk // compress_ratio)
    rows = q.shape[0]
    # No row scores beyond cdiv(max_seq_len, compress_ratio) compressed
    # columns. Round up to 64 to keep the logits row stride
    # cooperative_topk-compatible.
    logits_width = triton.cdiv(triton.cdiv(max_seq_len, compress_ratio), 64) * 64
    logits_width = min(max(64, logits_width), page_table.shape[1] * k_cache.shape[1])

    # chunk the inputs to keep temp logits below VLLM_SPARSE_INDEXER_MAX_LOGITS_MB
    max_logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
    rows_per_chunk = max(1, max_logits_bytes // (logits_width * 4))
    topk_workspace = torch.empty(
        (_TOPK_WORKSPACE_BYTES,), dtype=torch.uint8, device=q.device
    )

    # gather K cache into a contiguous buffer
    config, use_tma = _prefill_plan(page_table.shape[0])
    if use_tma:
        set_triton_allocator(q.device)
    gathered_k = k_cache.new_empty(page_table.shape[0], logits_width, q.shape[2])
    _gather_k_kernel[(page_table.shape[0], triton.cdiv(logits_width, 64))](
        k_cache,
        page_table,
        gathered_k,
        query_start_loc,
        visible_blocks,
        k_cache.stride(0),
        k_cache.stride(1),
        page_table.stride(0),
        gathered_k.stride(0),
        logits_width,
        PAGE_SIZE=k_cache.shape[1],
        HEAD_DIM=q.shape[2],
        CP=64,  # launch-bound floor; wider spills or idles SMs
        num_warps=8,
    )

    for query_start in range(0, rows, rows_per_chunk):
        query_end = min(query_start + rows_per_chunk, rows)
        query_slice = slice(query_start, query_end)
        logits = _prefill_logits(
            q,
            gathered_k,
            query_start_loc,
            visible_blocks,
            max_query_len,
            logits_width,
            query_offset=query_start,
            num_queries=query_end - query_start,
            config=config,
            use_tma=use_tma,
        )
        _topk(
            logits,
            visible_blocks[query_slice],
            token_topk,
            compress_ratio,
            block_indices[query_slice],
            topk_workspace,
        )


__all__ = [
    "expand_qsa_block_indices",
    "qsa_select_paged_decode",
    "qsa_select_paged_prefill",
    "warmup_qsa_mqa_prefill",
    "warmup_qsa_mqa_paged_decode",
]
