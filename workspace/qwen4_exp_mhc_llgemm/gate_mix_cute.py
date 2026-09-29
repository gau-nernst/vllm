# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CuteDSL hc_gate_mix prototype: flat 16B-vectorized elementwise kernel.

out[m, h] = (1/HC) * sum_s sigmoid(fp32(gate[m, s*HD + h])) * fp32(xn[m, s*HD + h])

Each thread owns VEC=8 contiguous output elements (never straddles a row since
HD % VEC == 0): 2*HC 16B loads, HC*VEC scalar sigmoid+FMA, one 16B bf16 store.
Grid is dynamic (ceil(M*HD/VEC / THREADS)) -- CUDA-graph capture fixes M per
graph, same as the triton original.
"""

import logging
from typing import Any

import cutlass
import cutlass.cute as cute
import torch
from cuda.bindings.driver import CUstream
from cutlass import const_expr

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


_LOG2E = 1.4426950408889634


def _sigmoid_f32(x, mode: str = "exp"):
    if mode == "identity":
        return x
    if mode == "tanh":
        # sigmoid(x) = 0.5 * (1 + tanh(x/2)); tanh.approx.f32 is a single SFU op
        return cute.math.tanh(x * 0.5, approx=True) * 0.5 + 0.5
    if mode == "exp2":
        return 1.0 / (1.0 + cute.math.exp2(x * (-_LOG2E), approx=True, ftz=True))
    return 1.0 / (1.0 + cute.math.exp(x * (-1.0)))


class GateMixCuteKernel:
    def __init__(
        self,
        hc: int = 4,
        hd: int = 2560,
        vec: int = 8,
        threads: int = 256,
        use_pdl: bool = True,
        sigmoid_mode: str = "exp",
    ):
        self.hc = hc
        self.hd = hd
        self.vec = vec
        self.threads = threads
        self.use_pdl = use_pdl
        self.sigmoid_mode = sigmoid_mode
        assert hd % vec == 0

    @cute.kernel
    def kernel(
        self,
        mG: cute.Tensor,  # [M, HC*HD] bf16
        mX: cute.Tensor,  # [M, HC*HD] bf16
        mO: cute.Tensor,  # [M, HD] bf16
        HC: cutlass.Constexpr,
        VEC: cutlass.Constexpr,
        THREADS: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        HD: cutlass.Constexpr = self.hd
        J: cutlass.Constexpr = HD // VEC

        if const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        t = bid * THREADS + tidx
        total = cute.size(mO, mode=[0]) * J
        if t < total:
            m = t // J
            j = t % J
            gG = cute.logical_divide(mG, (None, VEC))  # (M, (VEC, HC*J))
            gX = cute.logical_divide(mX, (None, VEC))
            gO = cute.logical_divide(mO, (None, VEC))  # (M, (VEC, J))

            g_frags, x_frags = [], []
            for s in cutlass.range_constexpr(HC):
                gf = cute.make_rmem_tensor_like(gG[m, (None, s * J + j)])
                cute.autovec_copy(gG[m, (None, s * J + j)], gf)
                g_frags.append(gf)
                xf = cute.make_rmem_tensor_like(gX[m, (None, s * J + j)])
                cute.autovec_copy(gX[m, (None, s * J + j)], xf)
                x_frags.append(xf)

            acc = cute.make_rmem_tensor((VEC,), cutlass.Float32)
            acc.fill(0.0)
            sig_mode: cutlass.Constexpr = self.sigmoid_mode
            for s in cutlass.range_constexpr(HC):
                g_f32 = g_frags[s].load().to(cutlass.Float32)
                x_f32 = x_frags[s].load().to(cutlass.Float32)
                for v in cutlass.range_constexpr(VEC):
                    acc[v] += _sigmoid_f32(g_f32[v], sig_mode) * x_f32[v]

            of = cute.make_rmem_tensor_like(gO[m, (None, j)])
            for v in cutlass.range_constexpr(VEC):
                of[v] = (acc[v] * (1.0 / HC)).to(cutlass.BFloat16)
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_launch_dependents()
            cute.autovec_copy(of, gO[m, (None, j)])

    @cute.jit
    def __call__(
        self,
        mG: cute.Tensor,
        mX: cute.Tensor,
        mO: cute.Tensor,
        stream: CUstream,
    ):
        total = cute.size(mO, mode=[0]) * (self.hd // self.vec)
        self.kernel(mG, mX, mO, self.hc, self.vec, self.threads).launch(
            grid=[cute.ceil_div(total, self.threads), 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )


class GateMixCute:
    def __init__(
        self,
        hc: int = 4,
        hd: int = 2560,
        vec: int = 8,
        threads: int = 256,
        use_pdl: bool = True,
        sigmoid_mode: str = "exp",
    ):
        self._hc = hc
        self._hd = hd
        self._vec = vec
        self._threads = threads
        self._use_pdl = use_pdl
        self._sigmoid_mode = sigmoid_mode
        self._compiled: Any = None

    def _compile(self) -> None:
        cute, _ = _cute()
        from cutlass import BFloat16
        from quack.compile_utils import make_fake_tensor

        m = cute.sym_int()
        g = make_fake_tensor(BFloat16, (m, self._hc * self._hd), divisibility=8)
        x = make_fake_tensor(BFloat16, (m, self._hc * self._hd), divisibility=8)
        o = make_fake_tensor(BFloat16, (m, self._hd), divisibility=8)
        kernel = GateMixCuteKernel(
            hc=self._hc,
            hd=self._hd,
            vec=self._vec,
            threads=self._threads,
            use_pdl=self._use_pdl,
            sigmoid_mode=self._sigmoid_mode,
        )
        self._compiled = cute.compile(
            kernel, g, x, o, _stream(), options="--enable-tvm-ffi"
        )

    def __call__(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        if self._compiled is None:
            self._compile()
        out = torch.empty(x.shape[0], self._hd, dtype=torch.bfloat16, device=x.device)
        self._compiled(gate, x, out, _stream())
        return out
