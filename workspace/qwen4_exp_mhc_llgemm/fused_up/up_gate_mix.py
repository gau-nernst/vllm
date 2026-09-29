# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Original workbench code. Dispatch/validation/compile-cache for the fused
mHC UP kernel (`_up_gate_mix.UpGateMixKernel`), following the conventions of
vllm/model_executor/kernels/linear/cute_dsl/ll_bf16.py.

Specialized to the production UP shape: K=320, N=10240, HC=4 -> out [M, 2560].
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

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


class UpGateMix:
    def __init__(
        self,
        *,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        max_m: int = 256,
        channels_per_cta: int = 4,
        channels_per_warp: int = 1,
        m_block: int = 8,
        m_splits: int = 1,
        lanes_per_dot: int = 32,
        impl: str = "warp",
        debug_mode: int = 0,
        tile_k: int = 64,
        num_stages: int = 3,
        ksplit: int = 1,
        min_blocks_per_mp: int = 1,
    ):
        self._k = k
        self._n = n
        self._hc = hc
        self._max_m = max_m
        self._ch = channels_per_cta
        self._cw = channels_per_warp
        self._mb = m_block
        self._m_splits = m_splits
        self._lpd = lanes_per_dot
        self._impl = impl
        self._debug_mode = debug_mode
        self._tile_k = tile_k
        self._num_stages = num_stages
        self._ksplit = ksplit
        self._min_blocks_per_mp = min_blocks_per_mp
        self._compiled: Any = None

    def _compile(self) -> None:
        cute, _ = _cute()
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        m = cute.sym_int()
        a = make_fake_tensor(BFloat16, (m, self._k), divisibility=8)
        w = make_fake_tensor(BFloat16, (self._n, self._k), divisibility=8)
        xn = make_fake_tensor(BFloat16, (m, self._n), divisibility=8)
        out = make_fake_tensor(BFloat16, (m, self._n // self._hc), divisibility=8)
        if self._impl == "toklane":
            from ._up_gate_mix_v2 import UpGateMixV2Kernel

            kernel = UpGateMixV2Kernel(
                k=self._k,
                n=self._n,
                hc=self._hc,
                max_m=self._max_m,
                channels_per_cta=self._ch,
                use_pdl=_use_pdl(),
                debug_mode=self._debug_mode,
            )
        elif self._impl == "v3":
            from ._up_gate_mix_v3 import UpGateMixV3Kernel

            kernel = UpGateMixV3Kernel(
                k=self._k,
                n=self._n,
                hc=self._hc,
                max_m=self._max_m,
                channels_per_cta=self._ch,
                channels_per_warp=self._cw,
                m_block=self._mb,
                m_splits=self._m_splits,
                lanes_per_dot=self._lpd,
                use_pdl=_use_pdl(),
                debug_mode=self._debug_mode,
            )
        elif self._impl == "v4":
            from ._up_gate_mix_v4 import UpGateMixV4Kernel

            kernel = UpGateMixV4Kernel(
                k=self._k,
                n=self._n,
                hc=self._hc,
                max_m=self._max_m,
                tile_k=self._tile_k,
                num_stages=self._num_stages,
                ksplit=self._ksplit,
                use_pdl=_use_pdl(),
                debug_mode=self._debug_mode,
                min_blocks_per_mp=self._min_blocks_per_mp,
            )
        else:
            from ._up_gate_mix import UpGateMixKernel

            kernel = UpGateMixKernel(
                k=self._k,
                n=self._n,
                hc=self._hc,
                max_m=self._max_m,
                channels_per_cta=self._ch,
                use_pdl=_use_pdl(),
            )
        self._compiled = cute.compile(
            kernel, a, w, xn, out, _stream(), options="--enable-tvm-ffi"
        )
        logger.debug("Compiled fused up_gate_mix kernel")

    def _validate_inputs(
        self, a: torch.Tensor, w: torch.Tensor, xn: torch.Tensor
    ) -> None:
        if a.dim() != 2 or w.dim() != 2 or xn.dim() != 2:
            raise ValueError("a, w, xn must be 2D tensors")
        if not (a.dtype == w.dtype == xn.dtype == torch.bfloat16):
            raise ValueError("a, w, xn must have dtype=bfloat16")
        if not (a.is_cuda and w.is_cuda and xn.is_cuda):
            raise ValueError("a, w, xn must be CUDA tensors")
        if a.device != w.device or a.device != xn.device:
            raise ValueError("a, w, xn must be on the same CUDA device")
        if a.shape[1] != self._k or w.shape != (self._n, self._k):
            raise ValueError(
                f"shape mismatch: a[M,{self._k}] w[{self._n},{self._k}] required"
            )
        if xn.shape != (a.shape[0], self._n):
            raise ValueError(f"xn must have shape [M, {self._n}]")
        if a.shape[0] > self._max_m:
            raise ValueError(f"fused up_gate_mix requires M <= {self._max_m}")
        # a may be a row-major strided view (e.g. the lora slice of the merged
        # hc_down_silu output, row stride 336): the kernels index a only
        # through the cute tensor and the compiled layout keeps the row stride
        # dynamic (divisibility=8).
        if a.stride(1) != 1 or a.stride(0) % 8 != 0:
            raise ValueError(
                "a must be row-major with a 16B-aligned row stride "
                "(multiple of 8 elements)"
            )
        if not (w.is_contiguous() and xn.is_contiguous()):
            raise ValueError("w, xn must be contiguous row-major")

    def __call__(
        self,
        a: torch.Tensor,  # [M, K] bf16 (silu'd lora)
        w: torch.Tensor,  # [N, K] bf16 (up projection weight)
        xn: torch.Tensor,  # [M, N] bf16 (normalized hidden states)
    ) -> torch.Tensor:  # [M, N // HC] bf16
        self._validate_inputs(a, w, xn)
        if self._compiled is None:
            self._compile()
        M = a.shape[0]
        out = torch.empty(M, self._n // self._hc, dtype=torch.bfloat16, device=a.device)
        self._compiled(a, w, xn, out, _stream())
        return out


up_gate_mix_kernel = UpGateMix()
up_gate_mix_kernel_v2 = UpGateMix(impl="toklane")
# v3 is SMEM-staged: max_m sets the activation staging footprint, which caps
# CTA occupancy. Keep a small-M variant (better occupancy) and a large-M one.
# channels_per_warp=2 register-blocks channels (each lane's activation
# fragment is reused across 2x4 weight fragments); m_block=4 keeps the
# CW*HC*MB=32 one-staging-row-per-lane invariant; m_splits=4 spreads M-blocks
# over 4 CTAs per channel group (phantom CTAs exit before the weight stream)
# to raise warp supply at M>=8. lanes_per_dot=16 splits each dot over 16
# lanes (2 dot groups/warp): halves the SMEM staging/reduce traffic at a 2x
# activation-load duplication cost -- measured faster at every M (6.30 vs
# 6.62 us at M=8, 4.51 vs 4.58 at M=1) and still bit-identical at M<=2.
_up_gate_mix_kernel_v3 = UpGateMix(
    impl="v3", max_m=32, channels_per_warp=2, m_block=4, m_splits=4, lanes_per_dot=16
)
_up_gate_mix_kernel_v3_xl = UpGateMix(
    impl="v3", max_m=64, channels_per_warp=2, m_block=4, m_splits=4, lanes_per_dot=16
)
# v4 (mma.sync): max_m sets the activation SMEM tile and accumulator count.
_up_gate_mix_kernel_v4 = UpGateMix(impl="v4", max_m=32)
_up_gate_mix_kernel_v4_xl = UpGateMix(impl="v4", max_m=64)


def up_gate_mix_gemm(
    a: torch.Tensor, w: torch.Tensor, xn: torch.Tensor
) -> torch.Tensor:
    return up_gate_mix_kernel(a, w, xn)


def up_gate_mix_gemm_v2(
    a: torch.Tensor, w: torch.Tensor, xn: torch.Tensor
) -> torch.Tensor:
    return up_gate_mix_kernel_v2(a, w, xn)


def up_gate_mix_gemm_v3(
    a: torch.Tensor, w: torch.Tensor, xn: torch.Tensor
) -> torch.Tensor:
    kernel = _up_gate_mix_kernel_v3 if a.shape[0] <= 32 else _up_gate_mix_kernel_v3_xl
    return kernel(a, w, xn)


def up_gate_mix_gemm_v4(
    a: torch.Tensor, w: torch.Tensor, xn: torch.Tensor
) -> torch.Tensor:
    kernel = _up_gate_mix_kernel_v4 if a.shape[0] <= 32 else _up_gate_mix_kernel_v4_xl
    return kernel(a, w, xn)


class UpGateMixV5:
    """tcgen05 variant: one compiled kernel per token bucket (mma_n in
    {8, 16, 32, 64}), selected by runtime M. M stays dynamic within a
    bucket (CUDA-graph safe: one graph per bucket). M > 64 uses ceil(M/mma_n)
    grid-y token tiles of the mma_n=32 (M <= 96) or mma_n=64 compilation;
    the weight stream is shared across tile CTAs via L2. Hard limits:
    M <= 512, mma_n <= 128 (4 stream accumulators x mma_n tmem cols
    <= 512)."""

    def __init__(
        self,
        *,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        mma_m: int = 64,
        bk: int = 64,
        num_w_stage: int = 16,
        use_pdl: bool | None = None,
        sigmoid_mode: str = "tanh",
        # None = per-bucket defaults {8:1, 16:2, 32:2, 64:4, 128:4} (swept).
        num_epilog_wg: int | None = None,
        mma_n_override: int | None = None,  # sweep knob: force the bucket
        debug_mode: int = 0,
        gemm_only: bool = False,  # plain [M, N] GEMM epilogue (no gate-mix)
        single_tma_warp: bool = False,  # merge activation TMA into warp 0
    ):
        self._k = k
        self._n = n
        self._hc = hc
        self._mma_m = mma_m
        self._bk = bk
        self._num_w_stage = num_w_stage
        self._use_pdl = _use_pdl() if use_pdl is None else use_pdl
        self._sigmoid_mode = sigmoid_mode
        self._num_epilog_wg = num_epilog_wg
        self._mma_n_override = mma_n_override
        self._debug_mode = debug_mode
        self._gemm_only = gemm_only
        self._single_tma_warp = single_tma_warp
        self._compiled: dict[tuple[int, bool], Any] = {}

    def _nwg(self, mma_n: int) -> int:
        if self._num_epilog_wg is not None:
            return max(1, min(self._num_epilog_wg, mma_n // 8))
        # Swept per-bucket defaults: 4 warpgroups win at mma_n=64 (M >= 64
        # incl. the tiled large-M path), 2 suffice at mma_n <= 32.
        nwg = {8: 1, 16: 2, 32: 2, 64: 4, 128: 4}[mma_n]
        return max(1, min(nwg, mma_n // 8))

    def _mark_a(self, a: torch.Tensor, strided: bool):
        from cutlass.cute.runtime import from_dlpack

        t = from_dlpack(a, assumed_align=16)
        if strided:
            # lora slice of the merged hc_down_silu output: row stride is
            # dynamic (e.g. 336 = 320 + 4 + 12 pad), M stays dynamic.
            return t.mark_layout_dynamic(leading_dim=1)
        return t.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)

    def _compile(self, mma_n: int, strided_a: bool = False) -> Any:
        from cutlass.cute import experimental as cute_ext
        from cutlass.cute.runtime import from_dlpack

        from ._up_gate_mix_v5 import UpGateMixV5Kernel

        kernel = UpGateMixV5Kernel(
            k=self._k,
            n=self._n,
            hc=self._hc,
            mma_m=self._mma_m,
            mma_n=mma_n,
            bk=self._bk,
            num_w_stage=self._num_w_stage,
            use_pdl=self._use_pdl,
            sigmoid_mode=self._sigmoid_mode,
            num_epilog_wg=self._nwg(mma_n),
            debug_mode=self._debug_mode,
            gemm_only=self._gemm_only,
            single_tma_warp=self._single_tma_warp,
        )
        dev = torch.device("cuda")
        w = torch.empty(self._n, self._k, dtype=torch.bfloat16, device=dev)
        a = torch.empty(mma_n, self._k, dtype=torch.bfloat16, device=dev)
        if strided_a:
            # Any 16B-aligned row stride works for the compile example; the
            # stride itself is dynamic. 336 = production merged down+inject.
            a = torch.empty(mma_n, 336, dtype=torch.bfloat16, device=dev)[:, : self._k]
        xn = torch.empty(mma_n, self._n, dtype=torch.bfloat16, device=dev)
        out_cols = self._n if self._gemm_only else self._n // self._hc
        out = torch.empty(mma_n, out_cols, dtype=torch.bfloat16, device=dev)
        compiled = cute_ext.compile(
            kernel,
            from_dlpack(w, assumed_align=16),
            self._mark_a(a, strided_a),
            from_dlpack(xn, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1), divisibility=1
            ),
            from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1), divisibility=1
            ),
            _stream(),
        )
        logger.debug("Compiled fused up_gate_mix v5 kernel (mma_n=%d)", mma_n)
        return compiled

    def __call__(
        self,
        a: torch.Tensor,  # [M, K] bf16
        w: torch.Tensor,  # [N, K] bf16
        xn: torch.Tensor,  # [M, N] bf16
    ) -> torch.Tensor:  # [M, N // HC] bf16
        from cutlass.cute.runtime import from_dlpack

        M = a.shape[0]
        if M > 512:
            raise ValueError("fused up_gate_mix v5 requires M <= 512")
        # Token bucket (tcgen05 N). M > 64 spills to ceil(M/mma_n) grid-y
        # token tiles (weight stream shared across tile CTAs via L2). One
        # wave = 152 CTAs = 40 channel tiles x <=3 token tiles; the swept
        # rule keeps every M <= 192 in one wave. mma_n=128 is intentionally
        # never auto-selected (heavier per-CTA epilogue; loses at all M).
        if self._mma_n_override is not None:
            mma_n = self._mma_n_override
        elif M <= 64:
            mma_n = 8 if M <= 8 else 16 if M <= 16 else 32 if M <= 32 else 64
        elif M <= 96:
            mma_n = 32  # 3 tiles = 120 CTAs, one wave; beats 2x64
        else:
            mma_n = 64  # ceil(M/64) tiles (one wave up to M=192)
        strided_a = a.stride(0) != a.shape[1]
        if strided_a and (a.stride(1) != 1 or a.stride(0) % 8 != 0):
            raise ValueError(
                "fused up_gate_mix v5 strided a must be row-major with a "
                "16B-aligned row stride (multiple of 8 elements)"
            )
        key = (mma_n, strided_a)
        if key not in self._compiled:
            self._compiled[key] = self._compile(mma_n, strided_a)
        out_cols = self._n if self._gemm_only else self._n // self._hc
        out = torch.empty(M, out_cols, dtype=torch.bfloat16, device=a.device)
        self._compiled[key](
            from_dlpack(w, assumed_align=16),
            self._mark_a(a, strided_a),
            from_dlpack(xn, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1), divisibility=1
            ),
            from_dlpack(out, assumed_align=16).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1), divisibility=1
            ),
            _stream(),
        )
        return out


_up_gate_mix_kernel_v5 = UpGateMixV5()


def up_gate_mix_gemm_v5(
    a: torch.Tensor, w: torch.Tensor, xn: torch.Tensor
) -> torch.Tensor:
    return _up_gate_mix_kernel_v5(a, w, xn)


class UpGateMixV6:
    """v6: standard-API rewrite of v5 (gn-kernels / bf16x3_router_gemm
    structure: atom TMA + descriptor prefetch, SmemAllocator, raw _tcgen05
    MMA/ld, make_fragment_C-style interleaved tmem stream packing). Same
    mma_n bucket dispatch as v5, but one compile per bucket covers both
    contiguous and strided a (row stride is a dynamic sym)."""

    def __init__(
        self,
        *,
        k: int = 320,
        n: int = 10240,
        hc: int = 4,
        mma_m: int = 64,
        bk: int = 64,
        num_w_stage: int = 16,
        use_pdl: bool | None = None,
        sigmoid_mode: str = "tanh",
        num_epilog_wg: int | None = None,
        mma_n_override: int | None = None,
        gemm_only: bool = False,  # plain GEMM epilogue: out is [M, N]
        debug_raw: int = -1,  # kernel debug knob (see UpGateMixV6Kernel)
    ):
        self._k = k
        self._n = n
        self._hc = hc
        self._mma_m = mma_m
        self._bk = bk
        self._num_w_stage = num_w_stage
        self._use_pdl = _use_pdl() if use_pdl is None else use_pdl
        self._sigmoid_mode = sigmoid_mode
        self._num_epilog_wg = num_epilog_wg
        self._mma_n_override = mma_n_override
        self._gemm_only = gemm_only
        self._debug_raw = debug_raw
        self._compiled: dict[int, Any] = {}

    def _nwg(self, mma_n: int) -> int:
        if self._num_epilog_wg is not None:
            return max(1, min(self._num_epilog_wg, mma_n // 8))
        nwg = {8: 1, 16: 2, 32: 2, 64: 4, 128: 4}[mma_n]
        return max(1, min(nwg, mma_n // 8))

    def _compile(self, mma_n: int) -> Any:
        import cutlass.cute as cute
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        from ._up_gate_mix_v6 import UpGateMixV6Kernel

        kernel = UpGateMixV6Kernel(
            k=self._k,
            n=self._n,
            hc=self._hc,
            mma_m=self._mma_m,
            mma_n=mma_n,
            bk=self._bk,
            num_w_stage=self._num_w_stage,
            use_pdl=self._use_pdl,
            sigmoid_mode=self._sigmoid_mode,
            num_epilog_wg=self._nwg(mma_n),
            gemm_only=self._gemm_only,
            debug_raw=self._debug_raw,
        )
        m = cute.sym_int()
        w = make_fake_tensor(BFloat16, (self._n, self._k), divisibility=8)
        # Dynamic row stride (divisibility=8): covers both contiguous a and
        # the strided lora slice (row stride 336) with one compile.
        a = make_fake_tensor(BFloat16, (m, self._k), divisibility=8)
        xn = make_fake_tensor(BFloat16, (m, self._n), divisibility=8)
        out_cols = self._n if self._gemm_only else self._n // self._hc
        out = make_fake_tensor(BFloat16, (m, out_cols), divisibility=8)
        compiled = cute.compile(
            kernel,
            w,
            a,
            xn,
            out,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        logger.debug("Compiled fused up_gate_mix v6 kernel (mma_n=%d)", mma_n)
        return compiled

    def __call__(
        self,
        a: torch.Tensor,  # [M, K] bf16
        w: torch.Tensor,  # [N, K] bf16
        xn: torch.Tensor,  # [M, N] bf16
    ) -> torch.Tensor:  # [M, N // HC] bf16
        M = a.shape[0]
        if M > 512:
            raise ValueError("fused up_gate_mix v6 requires M <= 512")
        if self._mma_n_override is not None:
            mma_n = self._mma_n_override
        elif M <= 64:
            mma_n = 8 if M <= 8 else 16 if M <= 16 else 32 if M <= 32 else 64
        elif M <= 96:
            mma_n = 32
        else:
            mma_n = 64
        if a.stride(1) != 1 or a.stride(0) % 8 != 0:
            raise ValueError(
                "fused up_gate_mix v6 requires row-major a with a 16B-aligned "
                "row stride (multiple of 8 elements)"
            )
        if mma_n not in self._compiled:
            self._compiled[mma_n] = self._compile(mma_n)
        out_cols = self._n if self._gemm_only else self._n // self._hc
        out = torch.empty(M, out_cols, dtype=torch.bfloat16, device=a.device)
        self._compiled[mma_n](w, a, xn, out)
        return out


_up_gate_mix_kernel_v6 = UpGateMixV6()


def up_gate_mix_gemm_v6(
    a: torch.Tensor, w: torch.Tensor, xn: torch.Tensor
) -> torch.Tensor:
    return _up_gate_mix_kernel_v6(a, w, xn)
