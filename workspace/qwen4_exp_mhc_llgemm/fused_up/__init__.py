# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Original workbench code: fused Qwen4Exp mHC UP projection + sigmoid gate
# mix (production semantics from vllm/models/qwen4_exp/nvidia/ops/hc.py).

from .up_gate_mix import (
    up_gate_mix_gemm,
    up_gate_mix_gemm_v2,
    up_gate_mix_gemm_v3,
    up_gate_mix_gemm_v4,
    up_gate_mix_gemm_v5,
)

__all__ = [
    "up_gate_mix_gemm",
    "up_gate_mix_gemm_v2",
    "up_gate_mix_gemm_v3",
    "up_gate_mix_gemm_v4",
    "up_gate_mix_gemm_v5",
]
