# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workspace copy of the production hc_down_silu dispatch, using the
early-launch_dependents kernel variants in this directory.

Only for PDL experiments: fires griddepcontrol.launch_dependents at CTA start
so a PDL-launched dependent (the fused up kernel) can overlap its weight
streaming with this kernel's body. Math is unchanged (bit-identical output).
"""

from functools import lru_cache
from typing import Any, Literal

import torch

MAX_FUSED_M = 48

# Same tuning as production hc_down_silu.py.
_SM100F_TUNED_SPLITK: dict[int, tuple[int, int, int]] = {
    **{m: (6, 5, 8) for m in range(5, 17)},
    **{m: (6, 5, 16) for m in range(17, 33)},
    **{m: (6, 4, 16) for m in range(33, MAX_FUSED_M + 1)},
}

CompileKey = tuple[Literal["fma", "fma_cap", "fma_cp", "mma"], ...]


class HcDownSiluGemmEarly:
    def __init__(
        self,
        rank: int,
        hc: int,
        k: int,
        *,
        prefetch_pdl_weights: bool = False,
        fma_cols_per_cta: int | None = None,
        fma_cpasync: tuple[int, int] | None = None,
    ) -> None:
        self.rank = rank
        self.hc = hc
        self.k = k
        self._prefetch_pdl_weights = prefetch_pdl_weights
        self._fma_cols_per_cta = fma_cols_per_cta
        self._fma_cpasync = fma_cpasync
        self._compiled: dict[CompileKey, Any] = {}

    def dispatch(self, m: int) -> CompileKey:
        if m <= 4 or self.k < 2048:
            if self._fma_cpasync is not None and self.k % 1024 == 0:
                j, s = self._fma_cpasync
                return ("fma_cp", m, self.k, 128, j, s)
            if self._fma_cols_per_cta is not None:
                return ("fma_cap", m, self.k, 128, self._fma_cols_per_cta)
            return ("fma", m, self.k, 128)
        return ("mma", *_SM100F_TUNED_SPLITK[m])

    def compile(self, compile_key: CompileKey) -> None:
        if compile_key in self._compiled:
            return

        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        N = cute.sym_int()
        gemm: Any
        extra_args: tuple[int, ...]
        if compile_key[0] == "fma_cp":
            from ._hc_down_silu_fma_cpasync import HcDownSiluFmaCpAsync

            _, M, K, threadblock_size, cols_per_cta, num_stages = compile_key
            gemm = HcDownSiluFmaCpAsync(
                k=K,
                cols_per_cta=cols_per_cta,
                num_stages=num_stages,
                threadblock_size=threadblock_size,
                rank=self.rank,
                hc=self.hc,
            )
            extra_args = (M, K, 1)  # runtime N placeholder
            options = "--enable-tvm-ffi"
        elif compile_key[0] == "fma_cap":
            from ._hc_down_silu_fma_capped import HcDownSiluFmaCapped

            _, M, K, threadblock_size, cols_per_cta = compile_key
            gemm = HcDownSiluFmaCapped(
                k=K,
                cols_per_cta=cols_per_cta,
                threadblock_size=threadblock_size,
                rank=self.rank,
                hc=self.hc,
            )
            extra_args = (M, K, 1)  # runtime N placeholder
            # The statically unrolled column loop needs register headroom for
            # ptxas to hoist next-column loads; occupancy no longer matters
            # (grid < #SMs by construction).
            options = "--enable-tvm-ffi --ptxas-options -maxrregcount=128"
        elif compile_key[0] == "mma":
            from ._hc_down_silu_mma_early import HcDownSiluMmaEarly

            _, split_k, num_stages, tile_n = compile_key
            M, K = cute.sym_int(), cute.sym_int()
            gemm = HcDownSiluMmaEarly(
                tile_n=tile_n,
                num_stages=num_stages,
                split_k=split_k,
                rank=self.rank,
                hc=self.hc,
            )
            extra_args = ()
            options = "--enable-tvm-ffi"
        else:
            from ._hc_down_silu_fma_early import HcDownSiluFmaEarly

            _, M, K, threadblock_size = compile_key
            gemm = HcDownSiluFmaEarly(
                k=K,
                threadblock_size=threadblock_size,
                prefetch_pdl_weights=self._prefetch_pdl_weights,
                rank=self.rank,
                hc=self.hc,
                # Cap residency at 3 CTAs/SM (nvvm.minctasm): at M=4 ptxas
                # otherwise allocates 220 regs/thread, only 2 CTAs/SM fit,
                # and the 324-CTA grid spills into a ragged second wave
                # (+1.3 us). The cap lowers M=4 to 168 regs with no spills.
                min_blocks_per_mp=3,
            )
            extra_args = (M, K, 1)  # runtime N placeholder
            options = "--enable-tvm-ffi --ptxas-options -maxrregcount=64"

        hidden_states = make_fake_tensor(BFloat16, (M, K), divisibility=8)
        router_weight = make_fake_tensor(BFloat16, (N, K), divisibility=8)
        output = make_fake_tensor(BFloat16, (M, N), divisibility=1)
        self._compiled[compile_key] = cute.compile(
            gemm,
            hidden_states,
            router_weight,
            output,
            *extra_args,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options=options,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,  # [M, K] bf16
        router_weight: torch.Tensor,  # [N, K] bf16
    ) -> torch.Tensor:  # [M, N] bf16
        M = hidden_states.shape[0]
        N = router_weight.shape[0]
        n_compute = self.rank + self.hc
        w_gemm = router_weight[:n_compute]
        compile_key = self.dispatch(M)
        self.compile(compile_key)
        kernel = self._compiled[compile_key]

        output = torch.empty(M, N, dtype=torch.bfloat16, device=hidden_states.device)
        out_gemm = output[:, :n_compute]
        if compile_key[0] == "mma":
            kernel(hidden_states, w_gemm, out_gemm)
        else:
            kernel(hidden_states, w_gemm, out_gemm, n_compute)
        return output


@lru_cache
def _get_kernel_early(
    rank: int,
    hc: int,
    k: int,
    prefetch_pdl_weights: bool = False,
    fma_cols_per_cta: int | None = None,
    fma_cpasync: tuple[int, int] | None = None,
) -> HcDownSiluGemmEarly:
    return HcDownSiluGemmEarly(
        rank,
        hc,
        k,
        prefetch_pdl_weights=prefetch_pdl_weights,
        fma_cols_per_cta=fma_cols_per_cta,
        fma_cpasync=fma_cpasync,
    )


def hc_down_silu_early(
    x: torch.Tensor,
    weight: torch.Tensor,
    rank: int,
    hc: int,
    fma_cols_per_cta: int | None = None,
    fma_cpasync: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Early-launch_dependents variant of production hc_down_silu.

    ``fma_cols_per_cta`` (M<=4 FMA path only) gives each CTA that many
    columns (grid = ceil(N/J), statically unrolled column loop), freeing SMs
    so a PDL-dependent up kernel co-schedules at t=0. ``fma_cpasync=(J, S)``
    is the same idea but with an S-stage cp.async smem pipeline so the
    capped grid still sustains HBM bandwidth. Both are bit-identical.
    """
    kernel = _get_kernel_early(
        rank,
        hc,
        weight.shape[1],
        x.shape[0] == 1,
        fma_cols_per_cta,
        fma_cpasync,
    )
    output = kernel(x, weight)
    return output[:, :rank], output[:, rank : rank + hc]
