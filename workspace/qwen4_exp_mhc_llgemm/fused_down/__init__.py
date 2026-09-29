# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from vllm/model_executor/kernels/linear/cute_dsl/ (ll_bf16 family)
# with the Qwen4Exp mHC SiLU epilogue fused into both backends.

from .ll_bf16_silu import ll_bf16_silu_gemm

__all__ = ["ll_bf16_silu_gemm"]
