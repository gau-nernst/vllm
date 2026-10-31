# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused NVFP4 gate_up GEMM + SwiGLU + NVFP4-quant kernel.

Reference is the production chain: FlashInfer cute-dsl mm_fp4 (bf16 out)
followed by torch.ops._C.silu_and_mul_nvfp4_quant. The fused kernel keeps
the GEMM accumulator in fp32 where the chain rounds to bf16, so its outputs
are compared by FP4 code distance (every code within one e2m1 step, bounded
mismatch fraction) and by e4m3 scale-byte distance, plus end-to-end through
the down_proj GEMM.
"""

import pytest
import torch

from vllm._custom_ops import create_fp4_scale_tensor, scaled_fp4_quant
from vllm.model_executor.kernels.linear.nvfp4.cute_dsl.gemm1_swiglu_nvfp4 import (
    gemm1_swiglu_nvfp4,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    interleave_gate_up_rows,
    pad_nvfp4_weight_for_cutlass,
    swizzle_blockscale,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm, has_flashinfer
from vllm.utils.math_utils import round_up

if not (
    current_platform.is_cuda()
    and current_platform.is_device_capability_family(100)
    and has_flashinfer()
):
    pytest.skip(
        "NVFP4 fused GEMM1+SwiGLU requires SM100+ and FlashInfer",
        allow_module_level=True,
    )

FP8_MAX = 448.0
FP4_MAX = 6.0

K = 5120
I_TP = [17408, 8704, 4352, 2176]  # TP1 / TP2 / TP4 / TP8 intermediate sizes
# Dispatch switches at M=128 (tile config) and M=512 (B evict-first); 300 has
# an odd number of 128-row scale blocks under the 256-row tile.
MS = [1, 2, 3, 7, 8, 16, 33, 64, 127, 128, 129, 256, 300, 512, 513, 1000, 2048, 8192]


def _seeded_bf16(m, n, seed, device="cuda"):
    """Realistic bf16 data: gaussian with a few large-magnitude channels."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, n), generator=g, device=device, dtype=torch.bfloat16)
    idx = torch.randperm(n, generator=g, device=device)[: n // 200 + 1]
    x[:, idx] *= 8.0
    return x


def _quant_x(x, gsf_inv, pad_m=False, poison=False):
    """Quantize activations through the production path; optionally pad the
    packed output rows to a multiple of 256 (Qwen3_5DenseMLP's bf16 input
    path). poison fills the rows past M, which no producer writes, with 0xFF.
    """
    m, k = x.shape
    m_pad = round_up(m, 256) if pad_m else m
    x_fp4 = torch.empty((m_pad, k // 2), dtype=torch.uint8, device=x.device)
    x_sf = create_fp4_scale_tensor(m_pad, k, x.device, True)
    if poison:
        x_fp4.fill_(0xFF)
        x_sf.view(torch.uint8).fill_(0xFF)
    torch.ops._C.scaled_fp4_quant.out(
        x,
        gsf_inv,
        True,
        output=x_fp4[:m],
        output_scale=x_sf[: round_up(m, 128)],
    )
    return x_fp4, x_sf.view(torch.float8_e4m3fn)


def _make_case(m, i, seed, pad_m=False, poison=False):
    x = _seeded_bf16(m, K, seed)
    w = _seeded_bf16(2 * i, K, seed + 1)
    x_gsf_inv = torch.abs(x).max().float() / (FP8_MAX * FP4_MAX)
    w_gsf_inv = torch.abs(w).max().float() / (FP8_MAX * FP4_MAX)
    x_fp4, x_sf = _quant_x(x, x_gsf_inv.reshape(1), pad_m, poison)
    w_fp4, w_sf = scaled_fp4_quant(
        w,
        w_gsf_inv.reshape(1),
        is_sf_swizzled_layout=False,
        backend="flashinfer-cutedsl",
    )
    alpha = (1.0 / x_gsf_inv / w_gsf_inv).float().reshape(1)
    return x_fp4, x_sf, w_fp4, w_sf, alpha


def _ref_chain(x_fp4, x_sf, w_fp4, w_sf, alpha, down_gsf_inv):
    """Production chain: cute-dsl mm_fp4 -> silu_and_mul_nvfp4_quant."""
    w_padded, _ = pad_nvfp4_weight_for_cutlass(w_fp4)
    w_sf_swz = swizzle_blockscale(w_sf)
    h = flashinfer_scaled_fp4_mm(
        x_fp4, w_padded, x_sf, w_sf_swz, alpha, torch.bfloat16, backend="cute-dsl"
    )
    m = h.shape[0]
    d = h.shape[1] // 2
    q = torch.empty((m, d // 2), dtype=torch.uint8, device=h.device)
    bs = torch.empty(
        (round_up(m, 128), round_up(d // 16, 4) // 4),
        dtype=torch.int32,
        device=h.device,
    ).view(torch.float8_e4m3fn)
    torch.ops._C.silu_and_mul_nvfp4_quant(q, bs, h, down_gsf_inv)
    return q, bs


def _fp4_codes(q):
    """Ordered integer per e2m1 code (sign-magnitude, +0 == -0)."""
    lo = (q & 0xF).to(torch.int16)
    hi = (q >> 4).to(torch.int16)
    c = torch.stack((lo, hi), dim=-1).reshape(q.shape[0], q.shape[1] * 2)
    return torch.where(c & 8 != 0, -(c & 7), c & 7)


def _e4m3_ord(b):
    """Ordered integer per e4m3 byte (sign-magnitude)."""
    bi = b.view(torch.uint8).to(torch.int16)
    return torch.where(bi & 0x80 != 0, -(bi & 0x7F), bi & 0x7F)


def _sf_dense(sf_swz, m, n_out):
    """Extract per-(row, 16-col group) SF bytes from the swizzled layout."""
    num_kt = n_out // 64
    kb = n_out // 16
    flat = sf_swz.view(torch.uint8).view(-1)
    r = torch.arange(m, device=flat.device)
    g = torch.arange(kb, device=flat.device)
    off = (
        (r[:, None] // 128 * num_kt + g[None, :] // 4) * 512
        + (r[:, None] % 32) * 16
        + ((r[:, None] // 32) % 4) * 4
        + g[None, :] % 4
    )
    return flat[off]


def _check_vs_chain(q, sf, q_ref, sf_ref, m, label):
    # Scale bytes: within one e4m3 step, bounded mismatch fraction.
    n_out = q.shape[1] * 2
    sa = _e4m3_ord(_sf_dense(sf, m, n_out))
    sb = _e4m3_ord(_sf_dense(sf_ref, m, n_out))
    sf_dist = (sa - sb).abs()
    assert (sf_dist > 1).sum().item() == 0, f"{label}: sf distance > 1"
    sf_frac = (sf_dist > 0).float().mean().item()
    assert sf_frac < 0.10, f"{label}: {sf_frac:.4%} sf bytes differ"

    # Every FP4 code within one e2m1 step; off-by-one fraction bounded. A
    # group whose scale underflows to 0 on one side only (fp32 vs bf16 amax
    # at the smallest e4m3 subnormal) quantizes to all zeros there, so its
    # codes are exempt; such groups must stay rare.
    dist = (_fp4_codes(q) - _fp4_codes(q_ref)).abs()
    underflow = (sa == 0) != (sb == 0)
    assert underflow.float().mean().item() < 1e-4, f"{label}: sf underflow"
    dist[underflow.repeat_interleave(16, dim=1)] = 0
    assert (dist > 1).sum().item() == 0, f"{label}: fp4 code distance > 1"
    frac_gt0 = (dist > 0).float().mean().item()
    assert frac_gt0 < 0.02, f"{label}: {frac_gt0:.4%} codes differ"


@pytest.mark.parametrize("i", I_TP)
@pytest.mark.parametrize("m", MS)
@torch.inference_mode()
def test_gemm1_swiglu_nvfp4_vs_chain(default_vllm_config, m: int, i: int):
    torch.set_default_device("cuda")
    x_fp4, x_sf, w_fp4, w_sf, alpha = _make_case(m, i, seed=1000 * i + m)
    down_gsf_inv = torch.tensor([0.5], dtype=torch.float32)

    q_ref, sf_ref = _ref_chain(x_fp4, x_sf, w_fp4, w_sf, alpha, down_gsf_inv)
    q, sf = gemm1_swiglu_nvfp4(
        x_fp4,
        x_sf,
        interleave_gate_up_rows(w_fp4),
        swizzle_blockscale(interleave_gate_up_rows(w_sf)),
        float(alpha),
        float(down_gsf_inv),
    )
    _check_vs_chain(q, sf, q_ref, sf_ref, m, f"m={m},i={i}")


@pytest.mark.parametrize("i", [17408])
@pytest.mark.parametrize("m", [1, 7, 33, 128, 129, 300, 1000])
@pytest.mark.parametrize("pad_m", [True, False])
@torch.inference_mode()
def test_gemm1_swiglu_nvfp4_padding(default_vllm_config, m: int, i: int, pad_m: bool):
    """A padded to 256 rows (bf16 input path) or exactly M rows (a
    pre-quantized QuantizedActivation); rows and scales past M hold 0xFF
    (NaN e4m3) and must not affect the stored outputs."""
    torch.set_default_device("cuda")
    x_fp4, x_sf, w_fp4, w_sf, alpha = _make_case(
        m, i, seed=7 * i + m, pad_m=pad_m, poison=True
    )
    down_gsf_inv = torch.tensor([0.5], dtype=torch.float32)

    q_ref, sf_ref = _ref_chain(x_fp4[:m], x_sf, w_fp4, w_sf, alpha, down_gsf_inv)
    q, sf = gemm1_swiglu_nvfp4(
        x_fp4,
        x_sf,
        interleave_gate_up_rows(w_fp4),
        swizzle_blockscale(interleave_gate_up_rows(w_sf)),
        float(alpha),
        float(down_gsf_inv),
        m=m,
    )
    _check_vs_chain(q, sf, q_ref, sf_ref, m, f"m={m},pad={pad_m}")


def _down_proj(q, sf, down_gsf_inv, m, i, k_out=5120, seed=0):
    """Run a dynamic-NVFP4 down_proj GEMM on a quantized activation."""
    w = _seeded_bf16(k_out, i, seed)
    w_gsf_inv = torch.abs(w).max().float() / (FP8_MAX * FP4_MAX)
    w_fp4, w_sf = scaled_fp4_quant(
        w,
        w_gsf_inv.reshape(1),
        is_sf_swizzled_layout=False,
        backend="flashinfer-cutedsl",
    )
    w_padded, _ = pad_nvfp4_weight_for_cutlass(w_fp4)
    w_sf_swz = swizzle_blockscale(w_sf)
    alpha = (1.0 / down_gsf_inv / w_gsf_inv).float().reshape(1)
    return flashinfer_scaled_fp4_mm(
        q, w_padded, sf, w_sf_swz, alpha, torch.bfloat16, backend="cute-dsl"
    )


@pytest.mark.parametrize("i", [17408, 8704])
@pytest.mark.parametrize("m", [1, 16, 127, 129, 2048])
@torch.inference_mode()
def test_gemm1_swiglu_nvfp4_through_down_proj(default_vllm_config, m: int, i: int):
    """The fused output feeds the down_proj GEMM as a QuantizedActivation."""
    torch.set_default_device("cuda")
    x_fp4, x_sf, w_fp4, w_sf, alpha = _make_case(m, i, seed=3 * i + m)
    down_gsf_inv = torch.tensor([0.5], dtype=torch.float32)

    q_ref, sf_ref = _ref_chain(x_fp4, x_sf, w_fp4, w_sf, alpha, down_gsf_inv)
    q, sf = gemm1_swiglu_nvfp4(
        x_fp4,
        x_sf,
        interleave_gate_up_rows(w_fp4),
        swizzle_blockscale(interleave_gate_up_rows(w_sf)),
        float(alpha),
        float(down_gsf_inv),
    )

    y_ref = _down_proj(q_ref, sf_ref, down_gsf_inv, m, i, seed=99)
    y = _down_proj(q, sf.view(torch.float8_e4m3fn), down_gsf_inv, m, i, seed=99)
    # The fused kernel keeps h in fp32 where the chain rounds to bf16,
    # flipping ~0.5% of fp4 codes and ~3% of scale bytes by one step; through
    # the K-wide down_proj reduction that lands at up to ~8% relative L2 per
    # row, below the fp4 quantization noise floor of the weights themselves
    # (~15% per element). The fused path is strictly more accurate (it skips
    # a rounding stage), this only measures the delta to the chain.
    err = (y.float() - y_ref.float()).norm(dim=1)
    ref_norm = y_ref.float().norm(dim=1).clamp_min(1e-6)
    assert (err / ref_norm).max().item() < 0.10


def _nvfp4_mlp(cls, hidden, i):
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config

    quant_config = ModelOptNvFp4Config(
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    return cls(
        hidden_size=hidden,
        intermediate_size=i,
        hidden_act="silu",
        quant_config=quant_config,
        prefix="model.layers.0.mlp",
    )


def _nvfp4_checkpoint(hidden, i, seed):
    """modelopt-style tensors per projection, gate and up with different
    global scales."""
    ckpt = {}
    for j, (name, n, k) in enumerate(
        (("gate_proj", i, hidden), ("up_proj", i, hidden), ("down_proj", hidden, i))
    ):
        w = _seeded_bf16(n, k, seed + j) * (1.0 + j)
        w_amax = torch.abs(w).max().float()
        w_fp4, w_sf = scaled_fp4_quant(
            w,
            (FP8_MAX * FP4_MAX / w_amax).reshape(1),
            is_sf_swizzled_layout=False,
            backend="flashinfer-cutedsl",
        )
        ckpt[name] = {
            "weight": w_fp4,
            "weight_scale": w_sf.view(torch.float8_e4m3fn),
            "weight_scale_2": (w_amax / (FP8_MAX * FP4_MAX)).reshape(()),
            "input_scale": torch.tensor(0.02 + 0.01 * j),
        }
    return ckpt


def _load(mlp, ckpt):
    # Same routing as the model's stacked_params_mapping.
    for name, shard in (("gate_proj", 0), ("up_proj", 1), ("down_proj", None)):
        layer = mlp.down_proj if shard is None else mlp.gate_up_proj
        for pname, t in ckpt[name].items():
            param = getattr(layer, pname)
            args = () if shard is None else (shard,)
            param.weight_loader(param, t, *args)


def _process(mlp):
    for layer in (mlp.gate_up_proj, mlp.down_proj):
        layer.quant_method.process_weights_after_loading(layer)


@torch.inference_mode()
def test_qwen3_5_dense_mlp_fused(default_vllm_config, dist_init):
    """Qwen3_5DenseMLP against the unfused Qwen2MoeMLP on the same
    checkpoint: the fused path (with its load-time 64-row gate/up
    interleave) for bf16 and QuantizedActivation inputs (exact or padded
    rows), and a layerwise weight reload."""
    from vllm.model_executor.layers.fusion.quant_activation import (
        QuantizedActivation,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kNvfp4Dynamic,
    )
    from vllm.model_executor.model_loader.reload import (
        finalize_layerwise_reload,
        initialize_layerwise_reload,
        record_metadata_for_reloading,
    )
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
    from vllm.models.qwen3_5.nvidia.mlp import Qwen3_5DenseMLP

    torch.set_default_device("cuda")
    hidden, i = 2048, 2176  # 17 gate/up blocks
    ckpt = _nvfp4_checkpoint(hidden, i, seed=11)
    ref = _nvfp4_mlp(Qwen2MoeMLP, hidden, i)
    mlp = _nvfp4_mlp(Qwen3_5DenseMLP, hidden, i)
    assert mlp.fused_gemm1
    record_metadata_for_reloading(mlp)
    for module in (ref, mlp):
        _load(module, ckpt)
        _process(module)

    def quantized(x, pad_rows):
        x_fp4, x_sf = _quant_x(
            x, mlp.gate_up_proj.input_global_scale_inv, pad_rows, poison=True
        )
        return QuantizedActivation(
            data=x_fp4,
            scale=x_sf.view(torch.float8_e4m3fn),
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=kNvfp4Dynamic,
        )

    for m in (37, 600):
        x = _seeded_bf16(m, hidden, seed=m)
        y_ref = ref(x)
        y = mlp(x)
        # The fused kernel keeps h in fp32 where the unfused chain rounds h
        # to bf16, so outputs agree only up to the quantization noise floor.
        err = (y.float() - y_ref.float()).norm(dim=1)
        assert (err / y_ref.float().norm(dim=1)).max().item() < 0.10
        for pad_rows in (False, True):
            torch.testing.assert_close(mlp(quantized(x, pad_rows)), y, rtol=0, atol=0)

    # A reload reprocesses the checkpoint weights in place (RL refit).
    weight_ptr = mlp.gate_up_proj.weight.data_ptr()
    initialize_layerwise_reload(mlp)
    _load(mlp, ckpt)
    finalize_layerwise_reload(mlp, None)
    assert mlp.gate_up_proj.weight.data_ptr() == weight_ptr
    torch.testing.assert_close(mlp(x), y, rtol=0, atol=0)
