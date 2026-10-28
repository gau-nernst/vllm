# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_qk_norm_rope import fused_qk_rmsnorm_rope_gate
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.rotary_embedding import (
    MRotaryEmbedding,
    RotaryEmbedding,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

# Qwen3.6 TP=1 attention geometry.
HEAD_DIM = 256
ROTARY_DIM = 64
RMS_NORM_EPS = 1e-6
MAX_POSITION_EMBEDDINGS = 262144
ROPE_THETA = 10000000.0
DTYPE = torch.bfloat16
SEED = 13
MROPE_SECTION = (11, 11, 10)
ROPE_CASES = [
    pytest.param(24, 4, None, id="rope"),
    pytest.param(16, 2, MROPE_SECTION, id="interleaved-mrope"),
]


@pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="fused_qk_rmsnorm_rope_gate Triton kernel requires CUDA/ROCm/XPU",
)
@pytest.mark.parametrize("num_q_heads,num_kv_heads,mrope_section", ROPE_CASES)
@pytest.mark.parametrize("num_tokens", [1, 4, 37])
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_matches_reference(
    default_vllm_config,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    mrope_section: tuple[int, int, int] | None,
) -> None:
    device = torch.device(
        current_platform.device_type, torch.accelerator.current_device_index()
    )
    torch.set_default_device(device)
    set_random_seed(SEED)

    q_gate = torch.randn(
        num_tokens, num_q_heads * 2 * HEAD_DIM, dtype=DTYPE, device=device
    )
    k = torch.randn(num_tokens, num_kv_heads * HEAD_DIM, dtype=DTYPE, device=device)

    q_norm = GemmaRMSNorm(HEAD_DIM, eps=RMS_NORM_EPS).to(device, dtype=DTYPE)
    k_norm = GemmaRMSNorm(HEAD_DIM, eps=RMS_NORM_EPS).to(device, dtype=DTYPE)
    q_norm.weight.normal_(std=0.1)
    k_norm.weight.normal_(std=0.1)

    q_gate_heads = q_gate.view(num_tokens, num_q_heads, 2 * HEAD_DIM)
    q = q_gate_heads[..., :HEAD_DIM]
    gate_ref = q_gate_heads[..., HEAD_DIM:].reshape(num_tokens, num_q_heads * HEAD_DIM)
    q_ref = q_norm.forward_native(q)
    k_ref = k_norm.forward_native(k.view(num_tokens, num_kv_heads, HEAD_DIM))
    assert isinstance(q_ref, torch.Tensor)
    assert isinstance(k_ref, torch.Tensor)
    q_ref = q_ref.reshape(num_tokens, num_q_heads * HEAD_DIM)
    k_ref = k_ref.reshape(num_tokens, num_kv_heads * HEAD_DIM)

    if mrope_section is None:
        rope = RotaryEmbedding(
            HEAD_DIM,
            ROTARY_DIM,
            MAX_POSITION_EMBEDDINGS,
            ROPE_THETA,
            True,
            DTYPE,
        ).to(device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
    else:
        rope = MRotaryEmbedding(
            HEAD_DIM,
            ROTARY_DIM,
            MAX_POSITION_EMBEDDINGS,
            ROPE_THETA,
            True,
            DTYPE,
            mrope_section=list(mrope_section),
            mrope_interleaved=True,
        ).to(device)
        positions = torch.arange(3 * num_tokens, dtype=torch.long, device=device).view(
            3, num_tokens
        )
        assert torch.unique(positions[:, 0]).numel() == 3

    q_ref, k_ref = rope.forward_native(positions, q_ref, k_ref)
    assert k_ref is not None

    q_out, k_out, gate_out = fused_qk_rmsnorm_rope_gate(
        q_gate,
        k,
        q_norm.weight,
        k_norm.weight,
        rope.cos_sin_cache,
        positions,
        RMS_NORM_EPS,
        num_q_heads,
        num_kv_heads,
        HEAD_DIM,
        ROTARY_DIM,
        mrope_section=mrope_section,
        norm_beta=1.0,
    )

    # The built-in reference performs RoPE in BF16, while the fused kernel
    # promotes the BF16-normalized values to FP32 for RoPE before storing BF16.
    torch.testing.assert_close(q_out, q_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(k_out, k_ref, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(gate_out, gate_ref, atol=0, rtol=0)


def _qk_norm_rope_inputs(num_tokens: int, num_q_heads: int, num_kv_heads: int):
    gen = torch.Generator(device="cuda").manual_seed(num_tokens)
    q_gate = torch.randn(
        num_tokens,
        num_q_heads * 2 * HEAD_DIM,
        dtype=DTYPE,
        device="cuda",
        generator=gen,
    )
    k = torch.randn(
        num_tokens, num_kv_heads * HEAD_DIM, dtype=DTYPE, device="cuda", generator=gen
    )
    q_weight = torch.randn(HEAD_DIM, dtype=DTYPE, device="cuda", generator=gen)
    k_weight = torch.randn(HEAD_DIM, dtype=DTYPE, device="cuda", generator=gen)
    cos_sin_cache = torch.randn(
        4 * num_tokens + 8, ROTARY_DIM, dtype=DTYPE, device="cuda", generator=gen
    )
    positions = torch.randint(
        0, 4 * num_tokens + 8, (num_tokens,), device="cuda", generator=gen
    )
    return q_gate, k, q_weight, k_weight, cos_sin_cache, positions


@pytest.mark.skipif(not current_platform.is_cuda(), reason="PDL is CUDA-only")
# 28 heads: 128 tokens (3584 blocks) is above the PDL grid gate.
@pytest.mark.parametrize("num_tokens", [1, 37, 128])
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_pdl_graph_bitwise(
    num_tokens: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Behind a producer in one CUDA graph, PDL on and off give bitwise equal
    outputs."""
    num_q_heads, num_kv_heads = 24, 4
    q_gate_in, k_in, q_weight, k_weight, cos_sin_cache, positions = (
        _qk_norm_rope_inputs(num_tokens, num_q_heads, num_kv_heads)
    )

    def run() -> tuple[torch.Tensor, ...]:
        q_gate, k = torch.empty_like(q_gate_in), torch.empty_like(k_in)

        def step():
            q_gate.copy_(q_gate_in)
            k.copy_(k_in)
            return fused_qk_rmsnorm_rope_gate(
                q_gate,
                k,
                q_weight,
                k_weight,
                cos_sin_cache,
                positions,
                RMS_NORM_EPS,
                num_q_heads,
                num_kv_heads,
                HEAD_DIM,
                ROTARY_DIM,
                norm_beta=1.0,
            )

        step()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = step()
        graph.replay()
        torch.accelerator.synchronize()
        return outs

    pdl_on = run()
    monkeypatch.setattr(current_platform, "is_arch_support_pdl", lambda: False)
    pdl_off = run()
    for on, off in zip(pdl_on, pdl_off):
        torch.testing.assert_close(on, off, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="PDL is CUDA-only")
@torch.inference_mode()
def test_fused_qk_norm_rope_gate_torch_compile_fullgraph() -> None:
    """Qwen3-Next traces this launcher under torch.compile(fullgraph=True);
    the PDL decision must not graph-break or add a token-count guard."""
    num_q_heads, num_kv_heads = 24, 4

    def fn(q_gate, k, q_weight, k_weight, cos_sin_cache, positions):
        return fused_qk_rmsnorm_rope_gate(
            q_gate,
            k,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            RMS_NORM_EPS,
            num_q_heads,
            num_kv_heads,
            HEAD_DIM,
            ROTARY_DIM,
            norm_beta=1.0,
        )

    compiled = torch.compile(fn, fullgraph=True, dynamic=True)
    for num_tokens in (3, 37, 128):
        inputs = _qk_norm_rope_inputs(num_tokens, num_q_heads, num_kv_heads)
        for out, ref in zip(compiled(*inputs), fn(*inputs)):
            torch.testing.assert_close(out, ref, rtol=0, atol=0)
