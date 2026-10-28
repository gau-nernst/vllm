# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_packed_decode,
)

DEVICE = current_platform.device_type

pytestmark = pytest.mark.skipif(
    not (current_platform.is_cuda_alike() or current_platform.is_xpu()),
    reason="Gated delta rule Triton kernels require a CUDA-alike or XPU device.",
)


@pytest.mark.parametrize("strided_mixed_qkv", [False, True])
@pytest.mark.parametrize(
    "batch,num_key_heads,num_value_heads,dtype,state_dtype",
    [
        (32, 4, 8, torch.float16, torch.float16),
        (32, 4, 8, torch.bfloat16, torch.bfloat16),
        (32, 4, 8, torch.float32, torch.float32),
        (1, 16, 64, torch.bfloat16, torch.float32),
        (9, 4, 12, torch.bfloat16, torch.float32),
        (1, 4, 12, torch.bfloat16, torch.bfloat16),
    ],
)
def test_fused_recurrent_packed_decode_matches_reference(
    dtype: torch.dtype,
    strided_mixed_qkv: bool,
    batch: int,
    num_key_heads: int,
    num_value_heads: int,
    state_dtype: torch.dtype,
):
    generator = torch.Generator(device=DEVICE).manual_seed(0)

    # Small but representative GDN config (Qwen3Next defaults are K=128, V=128).
    B = batch
    H = num_key_heads
    HV = num_value_heads
    K = 128
    V = 128
    qkv_dim = 2 * (H * K) + (HV * V)

    device = torch.device(DEVICE)

    if strided_mixed_qkv:
        # Simulate a packed view into a larger projection buffer:
        # mixed_qkv.stride(0) > mixed_qkv.shape[1]
        proj = torch.randn(
            (B, qkv_dim + 64), device=device, dtype=dtype, generator=generator
        )
        mixed_qkv = proj[:, :qkv_dim]
    else:
        mixed_qkv = torch.randn(
            (B, qkv_dim), device=device, dtype=dtype, generator=generator
        )

    a = torch.randn((B, HV), device=device, dtype=dtype, generator=generator)
    b = torch.randn((B, HV), device=device, dtype=dtype, generator=generator)
    A_log = torch.randn((HV,), device=device, dtype=dtype, generator=generator)
    dt_bias = torch.randn((HV,), device=device, dtype=dtype, generator=generator)

    # Continuous batching indices (include PAD_SLOT_ID=-1 cases). Index 0 is
    # reserved as NULL_BLOCK_ID (CUDA graph padding), so valid slots start at 1.
    ssm_state_indices = torch.arange(1, B + 1, device=device, dtype=torch.int32)
    if B > 3:
        ssm_state_indices[-3:] = -1
        ssm_state_indices[-1] = 0

    state0 = torch.randn(
        (B + 1, HV, V, K),
        device=device,
        dtype=state_dtype,
        generator=generator,
    )
    state_ref = state0.clone()
    state_packed = state0.clone()

    out_packed = torch.empty((B, 1, HV, V), device=device, dtype=dtype)

    # Reference path: materialize contiguous Q/K/V + explicit gating.
    q, k, v = torch.split(mixed_qkv, [H * K, H * K, HV * V], dim=-1)
    q = q.view(B, H, K).unsqueeze(1).contiguous()
    k = k.view(B, H, K).unsqueeze(1).contiguous()
    v = v.view(B, HV, V).unsqueeze(1).contiguous()

    x = a.float() + dt_bias.float()
    softplus_x = torch.where(
        x <= 20.0, torch.log1p(torch.exp(torch.clamp(x, max=20.0))), x
    )
    g = (-torch.exp(A_log.float()) * softplus_x).unsqueeze(1)
    beta = torch.sigmoid(b.float()).unsqueeze(1)

    out_ref, state_ref = fused_recurrent_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=K**-0.5,
        initial_state=state_ref,
        inplace_final_state=True,
        cu_seqlens=None,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    # Packed path: fused gating + recurrent directly from packed mixed_qkv.
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
    )

    atol = 2e-2 if dtype != torch.float32 else 1e-4
    rtol = 1e-2 if dtype != torch.float32 else 1e-4
    # Output rows for PAD_SLOT_ID entries are never written (uninitialized in
    # both paths), so compare only the valid rows.
    valid = ssm_state_indices > 0
    torch.testing.assert_close(out_packed[valid], out_ref[valid], rtol=rtol, atol=atol)
    torch.testing.assert_close(state_packed, state_ref, rtol=rtol, atol=atol)


def test_packed_decode_keeps_beta_in_fp32():
    device = torch.device(DEVICE)
    dtype = torch.bfloat16

    mixed_qkv = torch.ones((1, 3), device=device, dtype=dtype)
    a = torch.zeros((1, 1), device=device, dtype=dtype)
    b = torch.full((1, 1), 0.5, device=device, dtype=dtype)
    params = torch.zeros((1,), device=device, dtype=dtype)
    ssm_state_indices = torch.ones((1,), device=device, dtype=torch.int32)

    state_packed = torch.zeros((2, 1, 1, 1), device=device, dtype=torch.float32)
    out_packed = torch.empty((1, 1, 1, 1), device=device, dtype=dtype)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=state_packed,
        out=out_packed,
        ssm_state_indices=ssm_state_indices,
    )

    expected_beta = torch.sigmoid(b.float()).squeeze()
    torch.testing.assert_close(
        state_packed[1, 0, 0, 0], expected_beta, rtol=1e-6, atol=1e-6
    )


def test_packed_decode_supports_large_batch_head_grid():
    B, H, HV, K, V = 1024, 8, 64, 1, 1
    device = torch.device(DEVICE)
    gates = torch.empty((B, HV), device=device)
    params = torch.empty((HV,), device=device)
    out = torch.empty((B, 1, HV, V), device=device)

    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=torch.empty((B, 2 * H * K + HV * V), device=device),
        a=gates,
        b=gates,
        A_log=params,
        dt_bias=params,
        scale=1.0,
        initial_state=torch.empty((1, HV, V, K), device=device),
        out=out,
        ssm_state_indices=torch.zeros((B,), device=device, dtype=torch.int32),
    )

    assert torch.count_nonzero(out).item() == 0


@pytest.mark.parametrize("num_tokens", [1, 9])
@torch.inference_mode()
def test_packed_decode_strided_ba_bitwise(num_tokens: int) -> None:
    """Packed b/a chunk views (row stride 2*HV) must match contiguous inputs
    bitwise."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    device = torch.device(DEVICE)
    H, HV, K, V = 2, 4, 128, 128
    qkv_dim = 2 * H * K + HV * V

    qkvz = torch.randn(num_tokens, qkv_dim + HV * V, dtype=dtype, device=device)
    mixed_qkv = qkvz[:, :qkv_dim]
    ba = torch.randn(num_tokens, 2 * HV, dtype=dtype, device=device)
    b_view, a_view = ba.chunk(2, dim=-1)
    A_log = torch.randn(HV, dtype=torch.float32, device=device)
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device)
    state = torch.randn(num_tokens + 1, HV, V, K, dtype=torch.float32, device=device)
    indices = torch.arange(1, num_tokens + 1, dtype=torch.int32, device=device)

    out_strided = torch.empty(num_tokens, 1, HV, V, dtype=dtype, device=device)
    state_strided = state.clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a_view,
        b=b_view,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_strided,
        out=out_strided,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )

    out_contig = torch.empty(num_tokens, 1, HV, V, dtype=dtype, device=device)
    state_contig = state.clone()
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv.contiguous(),
        a=a_view.contiguous(),
        b=b_view.contiguous(),
        A_log=A_log,
        dt_bias=dt_bias,
        scale=K**-0.5,
        initial_state=state_contig,
        out=out_contig,
        ssm_state_indices=indices,
        use_qk_l2norm_in_kernel=True,
    )

    torch.testing.assert_close(out_strided, out_contig, rtol=0, atol=0)
    torch.testing.assert_close(state_strided, state_contig, rtol=0, atol=0)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="PDL launches are CUDA-only")
# Qwen3.8-27B TP1 GDN widths. B=64 is above the packed-decode and gated-norm
# PDL grid gates, so both launch without PDL there; B=40 is between them.
@pytest.mark.parametrize("batch", [1, 32, 40, 64])
@torch.inference_mode()
def test_packed_decode_pdl_chain_bitwise(
    batch: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Qwen3.5 decode chain conv1d update -> packed decode (early PDL
    trigger) -> gated RMS norm, replayed in one CUDA graph, gives bitwise the
    same outputs and states with PDL on and off."""
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.models.qwen3_5.nvidia.ops.fused_gdn_gated_norm import (
        fused_gdn_gated_rms_norm,
    )

    H, HV, K, V, width = 16, 48, 128, 128, 4
    dim = 2 * H * K + HV * V
    dtype = torch.bfloat16
    device = torch.device(DEVICE)
    gen = torch.Generator(device=device).manual_seed(batch)

    def randn(*shape, dtype=dtype):
        return torch.randn(*shape, generator=gen, device=device, dtype=dtype)

    qkvz_in = randn(batch, dim + HV * V)
    ba = randn(batch, 2 * HV)
    conv_state0 = randn(batch + 1, dim, width - 1)
    conv_weight = randn(dim, width) * 0.5
    conv_bias = randn(dim)
    A_log = randn(HV, dtype=torch.float32)
    dt_bias = randn(HV, dtype=torch.float32)
    state0 = randn(batch + 1, HV, V, K, dtype=torch.float32) * 0.1
    norm_weight = randn(V)
    indices = torch.arange(1, batch + 1, dtype=torch.int32, device=device)

    def run_chain() -> tuple[torch.Tensor, ...]:
        qkvz = torch.empty_like(qkvz_in)
        conv_state = conv_state0.clone()
        state = state0.clone()
        out = torch.empty(batch, 1, HV, V, dtype=dtype, device=device)
        b, a = ba.chunk(2, dim=-1)

        def step() -> None:
            qkvz.copy_(qkvz_in)
            mixed_qkv = causal_conv1d_update(
                qkvz[:, :dim],
                conv_state,
                conv_weight,
                conv_bias,
                "silu",
                conv_state_indices=indices,
            )
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=K**-0.5,
                initial_state=state,
                out=out,
                ssm_state_indices=indices,
                use_qk_l2norm_in_kernel=True,
            )
            out2d = out.view(batch, -1)
            fused_gdn_gated_rms_norm(
                x=out2d,
                z=qkvz[:, dim:],
                weight=norm_weight,
                out=out2d,
                eps=1e-6,
                num_v_heads=HV,
                head_v_dim=V,
                norm_before_gate=True,
                activation="silu",
            )

        step()  # compile outside capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        conv_state.copy_(conv_state0)
        state.copy_(state0)
        for _ in range(3):  # states carry across replays, like decode steps
            graph.replay()
        torch.accelerator.synchronize()
        return out, conv_state, state

    pdl_on = run_chain()
    monkeypatch.setattr(current_platform, "is_arch_support_pdl", lambda: False)
    pdl_off = run_chain()
    for on, off in zip(pdl_on, pdl_off):
        torch.testing.assert_close(on, off, rtol=0, atol=0)
