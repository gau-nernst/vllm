# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Top-k + top-p masking of BF16 logits for small k, with the FP32 cast fused.

Keeps the first k finite logits in (value desc, token id asc) order (all finite
ones if there are at most k), then, walking those in the same order with the
softmax over them, keeps a token while the probability mass ranked above it is
below p; everything else is -inf. That is the rule `apply_top_k_top_p_triton`
implements on `logits.float()`.

Logits are compared as 16-bit keys ordered like the BF16 values (-inf and NaN
map to 0). Each row is split into slices:
1. Every slice stores the maximum key of each 256-logit chunk.
2. The k-th largest chunk maximum is at most the k-th value, so every slice
   appends its logits at or above it to the row's candidates.
3. One program per row sorts the candidates and applies top-k and top-p. If
   too many logits tie at the bound, two key histograms find the k-th value.
4. Every slice writes -inf and the kept logits that fall in it.
"""

from typing import Any

import torch

from vllm.model_executor.warmup.jit_warmup import WarmupChoices
from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, triton

MAX_TOP_K = 64
_CHUNK = 256
_CAPACITY = 256
_NUM_SLICES = 4
_BLOCK = 1024


@triton.jit
def _orderable_key(bits):
    """BF16 bits -> int32 key in [0, 65535] with the order of the values;
    -inf and NaN map to 0, below every finite value."""
    u = bits.to(tl.int32) & 0xFFFF
    key = tl.where((u & 0x8000) != 0, (~u) & 0xFFFF, u | 0x8000)
    return tl.where((key >= 0x0080) & (key <= 0xFF80), key, 0)


@triton.jit
def _key_to_value(key):
    u = tl.where((key & 0x8000) != 0, key & 0x7FFF, (~key) & 0xFFFF)
    return (u << 16).to(tl.float32, bitcast=True)


@triton.jit
def _load_keys(row_ptr, offs, end):
    bits = tl.load(row_ptr + offs, mask=offs < end, other=float("-inf"))
    return _orderable_key(bits.to(tl.int16, bitcast=True))


@triton.jit
def _composite(key, token):
    """Orders (value desc, token id asc) under a descending sort."""
    return (key.to(tl.int64) << 32) | (0x7FFFFFFF - token).to(tl.int64)


@triton.jit
def _chunk_max_kernel(
    logits_ptr,
    logits_stride,
    chunk_max_ptr,
    vocab_size,
    slice_size,
    NUM_CHUNKS_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.program_id(1) * slice_size
    end = tl.minimum(start + slice_size, vocab_size)
    in_row = logits_ptr + row * logits_stride
    num_chunks = tl.cdiv(vocab_size, CHUNK)
    for block in tl.range(start, end, BLOCK, num_stages=4):
        key = _load_keys(in_row, block + tl.arange(0, BLOCK), end)
        chunk_max = tl.max(tl.reshape(key, (BLOCK // CHUNK, CHUNK)), axis=1)
        chunk_idx = block // CHUNK + tl.arange(0, BLOCK // CHUNK)
        tl.store(
            chunk_max_ptr + row * NUM_CHUNKS_PAD + chunk_idx,
            chunk_max,
            mask=chunk_idx < num_chunks,
        )


@triton.jit
def _gather_kernel(
    logits_ptr,
    logits_stride,
    chunk_max_ptr,
    candidates_ptr,
    num_candidates_ptr,
    k_ptr,
    vocab_size,
    slice_size,
    NUM_CHUNKS_PAD: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK: tl.constexpr,
    CHUNK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.program_id(1) * slice_size
    end = tl.minimum(start + slice_size, vocab_size)
    in_row = logits_ptr + row * logits_stride
    top_k = tl.load(k_ptr + row)
    chunk_idx = tl.arange(0, NUM_CHUNKS_PAD)
    chunk_max = tl.load(
        chunk_max_ptr + row * NUM_CHUNKS_PAD + chunk_idx,
        mask=chunk_idx < tl.cdiv(vocab_size, CHUNK),
        other=0,
    )
    chunk_max = tl.sort(chunk_max, descending=True)
    # With fewer than k finite chunk maxima, every finite logit is a candidate.
    bound = tl.maximum(tl.sum(tl.where(chunk_idx == top_k - 1, chunk_max, 0)), 1)
    for block in tl.range(start, end, BLOCK, num_stages=4):
        token = block + tl.arange(0, BLOCK)
        key = _load_keys(in_row, token, end)
        take = key >= bound
        slot = tl.atomic_add(num_candidates_ptr + row + 0 * token, 1, mask=take)
        tl.store(
            candidates_ptr + row * CAPACITY + slot,
            _composite(key, token),
            mask=take & (slot < CAPACITY),
        )


@triton.jit
def _kth_key_by_histogram(in_row, vocab_size, top_k, BLOCK: tl.constexpr):
    """Exact k-th largest key of a row with more than k finite logits."""
    bins = tl.arange(0, 256)
    hist_hi = tl.zeros((256,), dtype=tl.int32)
    for start in range(0, vocab_size, BLOCK):
        key = _load_keys(in_row, start + tl.arange(0, BLOCK), vocab_size)
        hist_hi += tl.histogram(key >> 8, 256, mask=key > 0)
    count_ge = tl.sum(hist_hi) - tl.cumsum(hist_hi, 0) + hist_hi
    hi = tl.sum((count_ge >= top_k).to(tl.int32)) - 1
    rank = top_k - tl.sum(tl.where(bins > hi, hist_hi, 0))
    hist_lo = tl.zeros((256,), dtype=tl.int32)
    for start in range(0, vocab_size, BLOCK):
        key = _load_keys(in_row, start + tl.arange(0, BLOCK), vocab_size)
        hist_lo += tl.histogram(key & 0xFF, 256, mask=(key > 0) & ((key >> 8) == hi))
    count_ge_lo = tl.sum(hist_lo) - tl.cumsum(hist_lo, 0) + hist_lo
    lo = tl.sum((count_ge_lo >= rank).to(tl.int32)) - 1
    return hi * 256 + lo


@triton.jit
def _select_kernel(
    logits_ptr,
    logits_stride,
    candidates_ptr,
    num_candidates_ptr,
    kept_ptr,
    num_kept_ptr,
    k_ptr,
    p_ptr,
    vocab_size,
    USE_TOP_P: tl.constexpr,
    CAPACITY: tl.constexpr,
    MAX_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    in_row = logits_ptr + row * logits_stride
    cand_row = candidates_ptr + row * CAPACITY
    top_k = tl.load(k_ptr + row)
    top_p = tl.load(p_ptr + row) if USE_TOP_P else 1.0
    num_cand = tl.load(num_candidates_ptr + row)
    if num_cand > CAPACITY:
        # Too many ties at the bound: find the k-th key exactly and collect
        # the logits above it plus its first copies by token id.
        key_k = _kth_key_by_histogram(in_row, vocab_size, top_k, BLOCK)
        num_greater = 0
        for start in range(0, vocab_size, BLOCK):
            key = _load_keys(in_row, start + tl.arange(0, BLOCK), vocab_size)
            num_greater += tl.sum((key > key_k).to(tl.int32))
        num_cand = 0
        num_ties = 0
        for start in range(0, vocab_size, BLOCK):
            token = start + tl.arange(0, BLOCK)
            key = _load_keys(in_row, token, vocab_size)
            tie = key == key_k
            tie_rank = num_ties + tl.cumsum(tie.to(tl.int32), 0) - 1
            take = (key > key_k) | (tie & (tie_rank < top_k - num_greater))
            slot = num_cand + tl.cumsum(take.to(tl.int32), 0) - 1
            tl.store(cand_row + slot, _composite(key, token), mask=take)
            num_cand += tl.sum(take.to(tl.int32))
            num_ties += tl.sum(tie.to(tl.int32))
        tl.debug_barrier()

    idx = tl.arange(0, CAPACITY)
    composite = tl.load(cand_row + idx, mask=idx < num_cand, other=-1)
    composite = tl.sort(composite, descending=True)
    is_cand = idx < tl.minimum(num_cand, top_k)
    value = _key_to_value((composite >> 32).to(tl.int32))
    row_max = tl.max(tl.where(is_cand, value, -1e30))
    prob = tl.where(is_cand, tl.exp(value - row_max), 0.0)
    prob = prob / tl.sum(prob)
    mass_above = tl.cumsum(prob, 0) - prob
    keep = is_cand & ((top_p >= 1.0) | (mass_above < top_p))
    # Keep the leading run, at most k <= MAX_K long: the parallel FP32 cumsum
    # is not monotone when p is within a few ulps of 1.
    num_kept = tl.min(tl.where(keep, CAPACITY, idx))
    tl.store(kept_ptr + row * MAX_K + idx, composite, mask=idx < num_kept)
    tl.store(num_kept_ptr + row, num_kept)


@triton.jit
def _write_kernel(
    out_ptr,
    out_stride,
    kept_ptr,
    num_kept_ptr,
    vocab_size,
    slice_size,
    MAX_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    start = tl.program_id(1) * slice_size
    end = tl.minimum(start + slice_size, vocab_size)
    out_row = out_ptr + row * out_stride
    for block in tl.range(start, end, BLOCK):
        offs = block + tl.arange(0, BLOCK)
        tl.store(out_row + offs, float("-inf"), mask=offs < end)
    tl.debug_barrier()
    idx = tl.arange(0, MAX_K)
    composite = tl.load(
        kept_ptr + row * MAX_K + idx, mask=idx < tl.load(num_kept_ptr + row), other=-1
    )
    token = 0x7FFFFFFF - (composite & 0x7FFFFFFF)
    value = _key_to_value((composite >> 32).to(tl.int32))
    in_slice = (composite >= 0) & (token >= start) & (token < end)
    tl.store(out_row + token, value, mask=in_slice)


def _slice_size(vocab_size: int) -> int:
    return triton.cdiv(triton.cdiv(vocab_size, _NUM_SLICES), _BLOCK) * _BLOCK


def _num_chunks_pad(vocab_size: int) -> int:
    return triton.next_power_of_2(triton.cdiv(vocab_size, _CHUNK))


def _chunk_max_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    logits_stride: Any = WarmupChoices(16, 2)
    return dict(
        logits=TritonWarmupTensor(
            torch.bfloat16, shape=(1, vocab_size), strides=(logits_stride, 1)
        ),
        chunk_max=TritonWarmupTensor(torch.int32),
    )


def _gather_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    logits_stride: Any = WarmupChoices(16, 2)
    return dict(
        logits=TritonWarmupTensor(
            torch.bfloat16, shape=(1, vocab_size), strides=(logits_stride, 1)
        ),
        chunk_max=TritonWarmupTensor(torch.int32),
        candidates=TritonWarmupTensor(torch.int64),
        num_candidates=TritonWarmupTensor(torch.int32),
        k=TritonWarmupTensor(torch.int32),
    )


def _select_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    logits_stride: Any = WarmupChoices(16, 2)
    use_top_p: Any = WarmupChoices(True, False)
    return dict(
        logits=TritonWarmupTensor(
            torch.bfloat16, shape=(1, vocab_size), strides=(logits_stride, 1)
        ),
        candidates=TritonWarmupTensor(torch.int64),
        num_candidates=TritonWarmupTensor(torch.int32),
        kept=TritonWarmupTensor(torch.int64),
        num_kept=TritonWarmupTensor(torch.int32),
        k=TritonWarmupTensor(torch.int32),
        p=TritonWarmupTensor(torch.float32) if use_top_p else None,
    )


def _write_warmup_inputs(vllm_config: Any) -> dict[str, Any]:
    vocab_size = vllm_config.model_config.get_vocab_size()
    return dict(
        out=TritonWarmupTensor(torch.float32, shape=(1, vocab_size)),
        kept=TritonWarmupTensor(torch.int64),
        num_kept=TritonWarmupTensor(torch.int32),
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_chunk_max_kernel, warmup_inputs=_chunk_max_warmup_inputs
)
def _chunk_max(logits: torch.Tensor, chunk_max: torch.Tensor) -> DispatchSpec:
    num_rows, vocab_size = logits.shape
    slice_size = _slice_size(vocab_size)
    return (num_rows, triton.cdiv(vocab_size, slice_size)), dict(
        vocab_size=vocab_size,
        slice_size=slice_size,
        NUM_CHUNKS_PAD=_num_chunks_pad(vocab_size),
        BLOCK=_BLOCK,
        CHUNK=_CHUNK,
        num_warps=4,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_gather_kernel, warmup_inputs=_gather_warmup_inputs
)
def _gather(
    logits: torch.Tensor,
    chunk_max: torch.Tensor,
    candidates: torch.Tensor,
    num_candidates: torch.Tensor,
    k: torch.Tensor,
) -> DispatchSpec:
    num_rows, vocab_size = logits.shape
    slice_size = _slice_size(vocab_size)
    return (num_rows, triton.cdiv(vocab_size, slice_size)), dict(
        vocab_size=vocab_size,
        slice_size=slice_size,
        NUM_CHUNKS_PAD=_num_chunks_pad(vocab_size),
        CAPACITY=_CAPACITY,
        BLOCK=_BLOCK,
        CHUNK=_CHUNK,
        num_warps=4,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_select_kernel, warmup_inputs=_select_warmup_inputs
)
def _select(
    logits: torch.Tensor,
    candidates: torch.Tensor,
    num_candidates: torch.Tensor,
    kept: torch.Tensor,
    num_kept: torch.Tensor,
    k: torch.Tensor,
    p: torch.Tensor | None,
) -> DispatchSpec:
    num_rows, vocab_size = logits.shape
    return (num_rows,), dict(
        p_ptr=k if p is None else p,
        vocab_size=vocab_size,
        USE_TOP_P=p is not None,
        CAPACITY=_CAPACITY,
        MAX_K=MAX_TOP_K,
        BLOCK=4096,
        num_warps=8,
    )


@triton_kernel_dispatcher_with_warmup(
    kernel=_write_kernel, warmup_inputs=_write_warmup_inputs
)
def _write(
    out: torch.Tensor, kept: torch.Tensor, num_kept: torch.Tensor
) -> DispatchSpec:
    num_rows, vocab_size = out.shape
    slice_size = _slice_size(vocab_size)
    return (num_rows, triton.cdiv(vocab_size, slice_size)), dict(
        vocab_size=vocab_size,
        slice_size=slice_size,
        MAX_K=MAX_TOP_K,
        BLOCK=_BLOCK,
        num_warps=4,
    )


def register_small_k_warmups() -> None:
    """Register the kernels of ``apply_top_k_top_p_small_k`` for JIT warmup."""
    _chunk_max.register_warmup()
    _gather.register_warmup()
    _select.register_warmup()
    _write.register_warmup()


def apply_top_k_top_p_small_k(
    logits: torch.Tensor, k: torch.Tensor, p: torch.Tensor | None
) -> torch.Tensor:
    """Masked FP32 copy of BF16 `logits` for per-row `k <= MAX_TOP_K`
    (`p` `None` or 1 disables top-p)."""
    assert logits.dtype == torch.bfloat16 and logits.stride(1) == 1
    num_rows, vocab_size = logits.shape
    device = logits.device
    out = torch.empty(num_rows, vocab_size, dtype=torch.float32, device=device)
    if num_rows == 0:
        return out
    k = k.to(torch.int32)
    if p is not None:
        p = p.to(torch.float32)
    chunk_max = torch.empty(
        num_rows, _num_chunks_pad(vocab_size), dtype=torch.int32, device=device
    )
    candidates = torch.empty(num_rows, _CAPACITY, dtype=torch.int64, device=device)
    num_candidates = torch.zeros(num_rows, dtype=torch.int32, device=device)
    kept = torch.empty(num_rows, MAX_TOP_K, dtype=torch.int64, device=device)
    num_kept = torch.empty(num_rows, dtype=torch.int32, device=device)
    _chunk_max(logits, chunk_max)
    _gather(logits, chunk_max, candidates, num_candidates, k)
    _select(logits, candidates, num_candidates, kept, num_kept, k, p)
    _write(out, kept, num_kept)
    return out
