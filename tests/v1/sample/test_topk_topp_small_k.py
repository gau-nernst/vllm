# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused small-k top-k/top-p kernel against the exact rule and the FP32
Triton path."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm.utils import jit_monitor
from vllm.v1.sample.ops.topk_topp_small_k import (
    MAX_TOP_K,
    _chunk_max,
    _gather,
    _select,
    _write,
    apply_top_k_top_p_small_k,
)
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA")


def _bf16_logits(num_rows: int, vocab_size: int, generator) -> torch.Tensor:
    """LLM-like rows: a Gaussian bulk plus a few strong tokens, on a coarse
    grid so that the k-th value and the top-p boundary often have ties."""
    logits = torch.randn(num_rows, vocab_size, device="cuda", generator=generator)
    strong = torch.randint(
        0, vocab_size, (num_rows, 8), device="cuda", generator=generator
    )
    boost = torch.rand(num_rows, 8, device="cuda", generator=generator) * 8 + 4
    logits.scatter_add_(1, strong, boost)
    return (torch.round(logits * 4) / 4).to(torch.bfloat16)


def _exact_top_k_top_p(logits: torch.Tensor, k, p) -> torch.Tensor:
    """The first k logits in (value desc, token id asc) order, then those whose
    softmax mass ranked above them is below p."""
    values, tokens = torch.sort(logits.float(), dim=1, descending=True, stable=True)
    values, tokens = values[:, :MAX_TOP_K], tokens[:, :MAX_TOP_K]
    is_top = torch.arange(MAX_TOP_K, device="cuda") < k[:, None]
    prob = torch.where(is_top, torch.exp(values - values[:, :1]), 0)
    prob = prob / prob.sum(1, keepdim=True)
    kept = is_top
    if p is not None:
        kept = kept & ((p[:, None] >= 1) | (prob.cumsum(1) - prob < p[:, None]))
    out = torch.full(logits.shape, float("-inf"), device="cuda")
    return out.scatter_(1, tokens, torch.where(kept, values, float("-inf")))


@pytest.mark.parametrize("num_rows", [48, 400])
@pytest.mark.parametrize("use_top_p", [True, False])
def test_matches_exact_top_k_top_p(num_rows: int, use_top_p: bool):
    generator = torch.Generator(device="cuda").manual_seed(num_rows)
    vocab_size = 248_320
    logits = _bf16_logits(num_rows, vocab_size, generator)
    k_choices = torch.tensor([1, 2, 5, 20, MAX_TOP_K], device="cuda")
    k = k_choices[
        torch.randint(0, 5, (num_rows,), device="cuda", generator=generator)
    ].to(torch.int32)
    p = None
    if use_top_p:
        p_choices = torch.tensor([0.1, 0.5, 0.9, 0.95, 1.0], device="cuda")
        p = p_choices[
            torch.randint(0, 5, (num_rows,), device="cuda", generator=generator)
        ]

    actual = apply_top_k_top_p_small_k(logits, k, p)

    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, _exact_top_k_top_p(logits, k, p), rtol=0, atol=0)
    # The FP32 Triton path agrees, except that on a few rows it keeps every
    # copy of the boundary value at the top-p cut (its tie search rebuilds the
    # value through log/exp and keeps all copies when that is not exact).
    triton_out = apply_top_k_top_p_triton(logits.float(), k, p)
    for row in (triton_out != actual).any(1).nonzero().flatten().tolist():
        extra = torch.isfinite(triton_out[row]) & ~torch.isfinite(actual[row])
        boundary = actual[row][torch.isfinite(actual[row])].min()
        assert not (
            torch.isfinite(actual[row]) & ~torch.isfinite(triton_out[row])
        ).any()
        assert (triton_out[row][extra] == boundary).all()


def test_top_p_near_one_keeps_a_leading_run():
    """With p a few ulps below 1 the parallel FP32 cumsum of the softmax can
    step back at the cut. The kept tokens must still be the leading run of the
    sorted top-k, and stale kept slots of an earlier call must not leak in."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    num_rows, vocab_size, top_k = 400, 248_320, 20
    k = torch.full((num_rows,), top_k, dtype=torch.int32, device="cuda")
    # Leave the caching allocator's kept buffer full of other rows' tokens.
    apply_top_k_top_p_small_k(
        _bf16_logits(num_rows, vocab_size, generator), k * 0 + MAX_TOP_K, None
    )
    # Peaked rows: a few close leaders, then top-k tails with probabilities
    # around one FP32 ulp of the accumulated mass.
    logits = torch.randn(num_rows, vocab_size, device="cuda", generator=generator)
    tokens = torch.randperm(vocab_size, device="cuda", generator=generator)[:top_k]
    leaders = 22 + torch.rand(num_rows, 4, device="cuda", generator=generator)
    tails = 12 - 15 * torch.rand(
        num_rows, top_k - 4, device="cuda", generator=generator
    )
    logits[:, tokens] = torch.cat([leaders, tails], dim=1)
    logits = logits.to(torch.bfloat16)
    p = torch.full((num_rows,), 1 - 2**-23, dtype=torch.float32, device="cuda")

    actual = apply_top_k_top_p_small_k(logits, k, p)

    values, order = torch.sort(logits.float(), dim=1, descending=True, stable=True)
    kept = torch.isfinite(actual.gather(1, order[:, :top_k]))
    num_kept = kept.sum(1, keepdim=True)
    assert torch.equal(kept, torch.arange(top_k, device="cuda") < num_kept)
    assert torch.equal(torch.isfinite(actual).sum(1, keepdim=True), num_kept)
    assert torch.equal(
        actual.gather(1, order[:, :top_k])[kept], values[:, :top_k][kept]
    )
    # The cut sits where the exact mass above crosses p, up to FP32 rounding.
    prob = torch.softmax(values[:, :top_k].double(), dim=1)
    mass_above = prob.cumsum(1) - prob
    assert (mass_above[kept] < p[0] + 1e-6).all()
    first_dropped = mass_above.gather(1, num_kept.clamp(max=top_k - 1))
    assert ((num_kept == top_k) | (first_dropped >= p[0] - 1e-6)).all()


def test_warmup_covers_runtime_launches():
    """The registered warmup compiles every specialization a call uses, so
    serving never JIT-compiles these kernels."""
    vocab_size = 248_320
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(get_vocab_size=lambda: vocab_size)
    )
    for kernel in (_chunk_max, _gather, _select, _write):
        # Drop what earlier tests compiled in this process.
        kernel.kernel.device_caches.clear()
        kernel.warmup(vllm_config=vllm_config)
    generator = torch.Generator(device="cuda").manual_seed(0)
    logits = _bf16_logits(64, vocab_size, generator)
    k = torch.full((64,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((64,), 0.95, dtype=torch.float32, device="cuda")

    jit_monitor.activate()
    with mock.patch.object(jit_monitor.logger, "warning_once") as warning:
        apply_top_k_top_p_small_k(logits, k, p)
        apply_top_k_top_p_small_k(logits, k, None)
        torch.accelerator.synchronize()
    warning.assert_not_called()
