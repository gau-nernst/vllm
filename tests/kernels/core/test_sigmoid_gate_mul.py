# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from vllm.model_executor.models.qwen2_moe import SigmoidGateMul
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_sigmoid_gate_mul(
    default_vllm_config,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    layer = SigmoidGateMul()
    z = torch.randn(num_tokens, d, dtype=dtype)
    x = torch.randn(num_tokens, d, dtype=dtype)
    weight = torch.randn(1, d, dtype=dtype)

    out = layer(z, x, weight)
    ref_out = layer.forward_native(z, x, weight)
    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )

    # Large logits exercise the sigmoid tails.
    weight = weight * 100
    out = layer(z, x, weight)
    ref_out = layer.forward_native(z, x, weight)
    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )

    # Row-strided inputs (every other row of wider buffers).
    z_base = torch.randn(2 * num_tokens, d, dtype=dtype)
    x_base = torch.randn(2 * num_tokens, d, dtype=dtype)
    out = layer(z_base[::2], x_base[::2], weight)
    ref_out = layer.forward_native(z_base[::2], x_base[::2], weight)
    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )
