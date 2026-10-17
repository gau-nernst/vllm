# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The fused small-k top-k/top-p kernel against the exact rule and the FP32
Triton path."""

import pytest
import torch

from vllm.v1.sample.ops.topk_topp_small_k import (
    MAX_TOP_K,
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
