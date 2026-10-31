# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense Qwen3.5 MLP with the fused NVFP4 gate_up + SwiGLU + quant GEMM.

Only the dense MLP uses this class. ``Qwen2MoeMLP`` stays the shared expert of
the MoE variants, where it runs on an aux stream beside the routed experts and
a persistent all-SM grid would contend with them.
"""

import torch

import vllm.envs as envs
from vllm._custom_ops import create_fp4_output_tensors
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.nvfp4.flashinfer import (
    FlashInferCuteDslNvFp4LinearKernel,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.fusion.quant_activation import (
    QuantizedActivation,
    as_quantized_activation,
    get_input_quant_key,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Dynamic
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up

logger = init_logger(__name__)

# The kernel's TMA box spans up to 256 activation rows; out-of-bounds rows
# cost ~2 us at decode, so quantize into a buffer padded to 256 rows.
_A_ROW_PAD = 256


class Qwen3_5DenseMLP(Qwen2MoeMLP):
    """Qwen2MoeMLP whose gate_up + SwiGLU + down_proj input quant run as one
    kernel on gate_up_proj's own weights (see gemm1_swiglu_nvfp4). The gate_up
    weight and scales are interleaved in 64-row [gate, up] blocks at load
    time; layers that do not qualify keep the unfused path unchanged."""

    def __init__(self, *args, enable_lora: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # LoRA would have to be applied to gate_up's output.
        self.fused_gemm1 = (
            not enable_lora
            and not envs.VLLM_QWEN3_5_DISABLE_FUSED_GEMM1
            and self._can_fuse_gemm1()
        )
        self._alpha: float | None = None
        self._down_gsf_inv: float | None = None
        if self.fused_gemm1:
            from vllm.model_executor.kernels.linear.nvfp4.cute_dsl.gemm1_swiglu_nvfp4 import (  # noqa: E501
                warmup_gemm1_swiglu_nvfp4,
            )

            # Read by FlashInferCuteDslNvFp4LinearKernel.process_weights_after_loading.
            self.gate_up_proj._gemm1_swiglu_nvfp4_interleave = True
            warmup_gemm1_swiglu_nvfp4(
                self.gate_up_proj.output_size_per_partition,
                self.gate_up_proj.input_size_per_partition,
                get_current_vllm_config().scheduler_config.max_num_batched_tokens,
            )
        logger.info_once(
            "Fused NVFP4 gate_up+SwiGLU+quant GEMM %s for dense MLP "
            "(2I=%d, K=%d per partition)",
            "enabled" if self.fused_gemm1 else "disabled",
            self.gate_up_proj.output_size_per_partition,
            self.gate_up_proj.input_size_per_partition,
            scope="process",
        )

    def _can_fuse_gemm1(self) -> bool:
        if not (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(100)
        ):
            return False
        from vllm.model_executor.kernels.linear.nvfp4.cute_dsl.gemm1_swiglu_nvfp4 import (  # noqa: E501
            gemm1_swiglu_nvfp4_supports,
        )

        gate_up = self.gate_up_proj
        # The fused kernel reads this kernel's packed weight and swizzled
        # scales; its output is the down_proj input of the unfused path
        # (maybe_fused_act_quant).
        kernel = getattr(gate_up.quant_method, "kernel", None)
        return (
            type(kernel) is FlashInferCuteDslNvFp4LinearKernel
            and get_input_quant_key(self.down_proj) == kNvfp4Dynamic
            and isinstance(self.act_fn, SiluAndMul)
            and self.expert_gate is None
            and gate_up.bias is None
            # Also rules out the N/K padding of pad_nvfp4_weight_for_cutlass.
            and gemm1_swiglu_nvfp4_supports(
                gate_up.output_size_per_partition, gate_up.input_size_per_partition
            )
        )

    def _quantize_padded(self, x: torch.Tensor) -> QuantizedActivation:
        """gate_up's input quant, into a buffer padded to _A_ROW_PAD rows."""
        assert x.dim() == 2
        m = x.shape[0]
        a_fp4, a_sf = create_fp4_output_tensors(
            round_up(m, _A_ROW_PAD), x.shape[1], x.device, is_sf_swizzled_layout=True
        )
        # The swizzled scales of the first round_up(m, 128) rows are a prefix
        # of the padded tensor.
        torch.ops._C.scaled_fp4_quant.out(
            x,
            self.gate_up_proj.input_global_scale_inv,
            True,
            output=a_fp4[:m],
            output_scale=a_sf[: round_up(m, 128)],
        )
        return QuantizedActivation(
            data=a_fp4,
            scale=a_sf,
            orig_dtype=x.dtype,
            orig_shape=x.shape,
            quant_key=kNvfp4Dynamic,
        )

    def forward(self, x: torch.Tensor | QuantizedActivation) -> torch.Tensor:
        if not self.fused_gemm1:
            return super().forward(x)
        from vllm.model_executor.kernels.linear.nvfp4.cute_dsl.gemm1_swiglu_nvfp4 import (  # noqa: E501
            gemm1_swiglu_nvfp4,
        )

        gate_up = self.gate_up_proj
        # Static per-layer scales; hoist the device->host read out of the
        # per-call path (and out of CUDA graph capture).
        if self._alpha is None:
            self._alpha = float(gate_up.alpha)
            self._down_gsf_inv = float(self.down_proj.input_global_scale_inv)
        alpha = self._alpha
        down_gsf_inv = self._down_gsf_inv
        assert alpha is not None and down_gsf_inv is not None
        # A pre-quantized input (from a fused norm) may carry padding rows
        # past orig_shape, which avoids the TMA out-of-bounds cost.
        qa = (
            self._quantize_padded(x)
            if isinstance(x, torch.Tensor)
            else as_quantized_activation(x, kNvfp4Dynamic)
        )
        assert qa is not None and len(qa.orig_shape) == 2
        m = qa.orig_shape[0]
        q, sf = gemm1_swiglu_nvfp4(
            qa.data,
            qa.scale,
            gate_up.weight,
            gate_up.weight_scale,
            alpha,
            down_gsf_inv,
            m=m,
        )
        out, _ = self.down_proj(
            QuantizedActivation(
                data=q,
                scale=sf.view(torch.float8_e4m3fn),
                orig_dtype=qa.orig_dtype,
                orig_shape=torch.Size((m, q.shape[1] * 2)),
                quant_key=kNvfp4Dynamic,
            )
        )
        return out
