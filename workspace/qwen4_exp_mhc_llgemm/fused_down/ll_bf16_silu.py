# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Adapted from vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py.

Dispatch/validation/warmup for the mHC-SiLU-fused DOWN GEMM (K=10240, N=336,
SiLU(x/HC) on columns < rank). Output is bf16 (ll_bf16's fp32-output bonus is
given up to preserve the production rounding boundary).
"""

import logging
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

logger = logging.getLogger(__name__)

# Default configs (same as ll_bf16.py)
_DEFAULT_DOTPROD_BS = 128
_DEFAULT_DOTPROD_MAX_M = 4
_DEFAULT_SPLITK_CONFIG = (6, 4)  # (split_k, num_stages); tile_n is per-instance

# SM100f-specific tuned configs. Split-K values are (split_k, num_stages,
# tile_n) tuples; the mHC DOWN shape entries (K=10240, N_compute=324) were
# swept on GB300 in this workspace (see RESULTS_FUSED_DOWN_OPT.md).
_SM100F_TUNED_DOTPROD_BS: dict[tuple[int, int], dict[int, int]] = {
    (6144, 256): {M: 256 for M in (1, 3, 4)},
    # NOTE: the mHC DOWN shape (10240, 324) deliberately keeps bs=128: bs=256
    # is ~0.1-0.35 us faster but changes the shuffle reduction tree, which
    # breaks bit-identity with the production chain on some data (1-ulp flips
    # observed at M=4). See RESULTS_FUSED_DOWN_OPT.md.
}
_SM100F_TUNED_SPLITK_CONFIGS: dict[tuple[int, int], dict[int, tuple[int, int, int]]] = {
    (4096, 256): {
        **{M: (8, 5, 16) for M in (5, 8)},
        9: (8, 2, 16),
    },
    (7168, 256): {14: (8, 2, 16)},
    (6144, 256): {M: (8, 2, 16) for M in (9, 12, 16)},
    (7168, 384): {M: (7, 5, 16) for M in (13, 16)},
    # (6, 5, tile_n) keeps split_k=6, i.e. the same K reduction order as the
    # production chain -> stays bit-identical. num_stages/tile_n only affect
    # pipelining and N-tiling.
    (10240, 324): {8: (6, 5, 8), 16: (6, 5, 8), 32: (6, 5, 16)},
}


def _arch_tuned_configs():
    from vllm.platforms import current_platform

    if current_platform.is_device_capability_family(100):
        return (
            _SM100F_TUNED_DOTPROD_BS,
            _SM100F_TUNED_SPLITK_CONFIGS,
        )
    return {}, {}


_cute_ctx = None


def _cute():
    global _cute_ctx
    if _cute_ctx is not None:
        return _cute_ctx
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream

    _cute_ctx = (cute, CUstream)
    return _cute_ctx


def _stream():
    _, CUstream = _cute()
    from vllm.utils.torch_utils import current_stream

    return CUstream(current_stream().cuda_stream)


def _use_pdl() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_arch_support_pdl()


class LLBf16SiluGemm:
    @dataclass(frozen=True, slots=True)
    class CompileKey:
        backend: Literal["dotprod", "splitk"]
        M: int = 0
        K: int = 0
        bs: int = 0
        split_k: int = 0
        num_stages: int = 0
        tile_n: int = 0  # 0 = use the instance default

    def __init__(
        self,
        *,
        prefetch_pdl_weights: bool = False,
        prefetch_tiles: int | None = None,
        n_compute: int | None = None,
        rank: int = 320,
        hc: int = 4,
        dotprod_bs: int | None = None,
        main_vec_width: int = 8,
        tail_vec_width: int = 4,
        splitk_config: tuple[int, int] | None = None,  # (split_k, num_stages)
        splitk_static_k: bool = False,
        tile_n: int = 16,  # also the tuned-table default tile_n
        tile_k: int = 256,
        num_dma_warps: int = 4,
    ) -> None:
        self._prefetch_pdl_weights = prefetch_pdl_weights
        self._prefetch_tiles = prefetch_tiles
        # Compute only the first ``n_compute`` columns of the [M, N] output;
        # the rest is left uninitialized. Production discards the pad columns
        # (N = rank + hc + 12), so n_compute=rank+hc skips the pad weight rows.
        self._n_compute = n_compute
        self._rank = rank
        self._hc = hc
        # Config overrides (instance-wide; per-instance caches make the
        # existing cache keys sufficient).
        self._dotprod_bs = dotprod_bs
        self._main_vec_width = main_vec_width
        self._tail_vec_width = tail_vec_width
        self._splitk_config = splitk_config
        self._splitk_static_k = splitk_static_k
        self._tile_n = tile_n
        self._tile_k = tile_k
        self._num_dma_warps = num_dma_warps
        # Dot-prod: keyed on (M, K, bs), because M and K are Constexpr.
        self._compiled_cache: dict[tuple[int, int, int], Any] = {}
        # Split-K: keyed on (split_k, num_stages), fully shape-dynamic.
        self._splitk_cache: dict[tuple[int, int], Any] = {}

    def dispatch(self, *, M: int, K: int, N: int) -> CompileKey:
        tuned_bs, tuned_splitk = _arch_tuned_configs()
        if M <= _DEFAULT_DOTPROD_MAX_M or K < 2048:
            bs = self._dotprod_bs
            if bs is None:
                bs = tuned_bs.get((K, N), {}).get(M, _DEFAULT_DOTPROD_BS)
            return self.CompileKey(backend="dotprod", M=M, K=K, bs=bs)

        if self._splitk_config is not None:
            split_k, num_stages = self._splitk_config
            tile_n = self._tile_n
        else:
            split_k, num_stages, tile_n = tuned_splitk.get((K, N), {}).get(
                M, (*_DEFAULT_SPLITK_CONFIG, self._tile_n)
            )
        return self.CompileKey(
            backend="splitk",
            K=K if self._splitk_static_k else 0,
            split_k=split_k,
            num_stages=num_stages,
            tile_n=tile_n,
        )

    def _fake_gemm_tensors(self, *, M, K, N, divisibility: int):
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        hidden_states = make_fake_tensor(BFloat16, (M, K), divisibility=divisibility)
        router_weight = make_fake_tensor(BFloat16, (N, K), divisibility=divisibility)
        output = make_fake_tensor(BFloat16, (M, N), divisibility=1)
        return hidden_states, router_weight, output

    def _compile_splitk(self, compile_key: CompileKey) -> None:
        cute, _ = _cute()
        from ._splitk_silu import LLBf16SiluSplitK

        # K=0 in the compile key means shape-dynamic K; otherwise specialize.
        k_shape = compile_key.K if compile_key.K > 0 else cute.sym_int()
        hidden_states, router_weight, output = self._fake_gemm_tensors(
            M=cute.sym_int(),
            K=k_shape,
            N=cute.sym_int(),
            divisibility=8,
        )
        gemm = LLBf16SiluSplitK(
            tile_n=compile_key.tile_n or self._tile_n,
            tile_k=self._tile_k,
            num_stages=compile_key.num_stages,
            num_dma_warps=self._num_dma_warps,
            split_k=compile_key.split_k,
            use_pdl=_use_pdl(),
            static_k=compile_key.K > 0,
            rank=self._rank,
            hc=self._hc,
        )
        compiled = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            _stream(),
            options="--enable-tvm-ffi",
        )
        self._splitk_cache[
            (
                compile_key.split_k,
                compile_key.num_stages,
                compile_key.tile_n,
                compile_key.K,
            )
        ] = compiled
        logger.debug(
            "Compiled ll_bf16_silu_splitk: sk=%d ns=%d",
            compile_key.split_k,
            compile_key.num_stages,
        )

    def _compile_dotprod(self, compile_key: CompileKey) -> None:
        cute, _ = _cute()
        from ._dotprod_silu import LLBf16SiluDotprod

        N = cute.sym_int()
        stride_divisibility = math.gcd(8, compile_key.K)
        hidden_states, router_weight, output = self._fake_gemm_tensors(
            M=compile_key.M,
            K=compile_key.K,
            N=N,
            divisibility=stride_divisibility,
        )
        gemm = LLBf16SiluDotprod(
            k=compile_key.K,
            bs=compile_key.bs,
            main_vec_width=self._main_vec_width,
            tail_vec_width=self._tail_vec_width,
            use_pdl=_use_pdl(),
            prefetch_pdl_weights=self._prefetch_pdl_weights,
            prefetch_tiles=self._prefetch_tiles,
            rank=self._rank,
            hc=self._hc,
        )
        compiled = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            compile_key.M,
            compile_key.K,
            1,  # runtime N placeholder for fake-tensor compile
            _stream(),
            options="--enable-tvm-ffi --ptxas-options -maxrregcount=64",
        )
        self._compiled_cache[(compile_key.M, compile_key.K, compile_key.bs)] = compiled
        logger.debug(
            "Compiled ll_bf16_silu_dotprod: M=%d, K=%d, bs=%d",
            compile_key.M,
            compile_key.K,
            compile_key.bs,
        )

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key.backend == "splitk":
            splitk_cache_key = (
                compile_key.split_k,
                compile_key.num_stages,
                compile_key.tile_n,
                compile_key.K,
            )
            if splitk_cache_key not in self._splitk_cache:
                self._compile_splitk(compile_key)
            return

        dotprod_cache_key = (compile_key.M, compile_key.K, compile_key.bs)
        if dotprod_cache_key not in self._compiled_cache:
            self._compile_dotprod(compile_key)

    @staticmethod
    def _validate_inputs(
        hidden_states: torch.Tensor,
        router_weight: torch.Tensor,
    ) -> None:
        if hidden_states.dim() != 2 or router_weight.dim() != 2:
            raise ValueError("hidden_states and router_weight must be 2D tensors")
        if (
            hidden_states.dtype != torch.bfloat16
            or router_weight.dtype != torch.bfloat16
        ):
            raise ValueError("hidden_states and router_weight must have dtype=bfloat16")
        if hidden_states.device.type != "cuda" or router_weight.device.type != "cuda":
            raise ValueError(
                "hidden_states and router_weight must have device_type=cuda"
            )
        if hidden_states.device != router_weight.device:
            raise ValueError(
                "hidden_states and router_weight must be on the same CUDA device"
            )
        if hidden_states.shape[1] != router_weight.shape[1]:
            raise ValueError(
                "hidden_states and router_weight must have matching K dimensions"
            )
        # Kernels use vectorized bf16 loads and require 16-byte row alignment.
        if hidden_states.shape[1] % 8 != 0:
            raise ValueError("ll_bf16_silu_gemm requires K to be divisible by 8")
        if not hidden_states.is_contiguous() or not router_weight.is_contiguous():
            raise ValueError("ll_bf16_silu_gemm requires contiguous row-major inputs")

    def __call__(
        self,
        hidden_states: torch.Tensor,  # [M, K] bf16
        router_weight: torch.Tensor,  # [N, K] bf16
    ) -> torch.Tensor:  # [M, N] bf16
        self._validate_inputs(hidden_states, router_weight)

        M, K = hidden_states.shape
        N = router_weight.shape[0]
        n_compute = N
        if self._n_compute is not None and self._n_compute < N:
            n_compute = self._n_compute
        w_gemm = router_weight[:n_compute]
        compile_key = self.dispatch(M=M, K=K, N=n_compute)
        if compile_key.backend == "splitk":
            splitk_cache_key = (
                compile_key.split_k,
                compile_key.num_stages,
                compile_key.tile_n,
                compile_key.K,
            )
            if splitk_cache_key not in self._splitk_cache:
                self.compile(compile_key)
            kernel = self._splitk_cache[splitk_cache_key]
        else:
            dotprod_cache_key = (compile_key.M, compile_key.K, compile_key.bs)
            if dotprod_cache_key not in self._compiled_cache:
                self.compile(compile_key)
            kernel = self._compiled_cache[dotprod_cache_key]

        stream = _stream()
        output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
        out_gemm = output[:, :n_compute] if n_compute < N else output
        if compile_key.backend == "splitk":
            kernel(hidden_states, w_gemm, out_gemm, stream, 1.0)
        else:
            kernel(hidden_states, w_gemm, out_gemm, n_compute, stream)
        return output


ll_bf16_silu_gemm_kernel = LLBf16SiluGemm(n_compute=324)
ll_bf16_silu_gemm_c1_pdl_kernel = LLBf16SiluGemm(
    prefetch_pdl_weights=True, n_compute=324
)


def ll_bf16_silu_gemm(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> torch.Tensor:
    kernel = (
        ll_bf16_silu_gemm_c1_pdl_kernel
        if hidden_states.shape[0] == 1
        else ll_bf16_silu_gemm_kernel
    )
    return kernel(hidden_states, router_weight)
