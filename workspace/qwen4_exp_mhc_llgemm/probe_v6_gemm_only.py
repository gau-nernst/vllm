# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GEMM-only base comparison: v6 (raw gn-kernels style) vs v5 vs nvjet.

Per M, each captured alone in a CUDA graph with cold-L2 CUPTI timing
(PDL off, standalone):

  nvjet       F.linear(a[M,320], w[10240,320]) -> [M,10240] (cuBLAS)
  v5_gemm     v5 tcgen05 GEMM-only epilogue -> [M,10240] bf16 store
  v6_gemm     v6 raw tcgen05 GEMM-only epilogue -> [M,10240] bf16 store
  v6_dm200    v6 debug_raw=200: TMA+MMA only, no tmem readback/store

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/probe_v6_gemm_only.py [M_LIST]
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import (
    UpGateMixV5,
    UpGateMixV6,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
K, N = 320, 10240
M_LIST = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST
    flush_buf = torch.empty(
        2 * get_l2_cache_size(DEVICE), device=DEVICE, dtype=torch.int8
    )

    v5_gemm = UpGateMixV5(use_pdl=False, gemm_only=True)
    v6_gemm = UpGateMixV6(use_pdl=False, gemm_only=True)

    for m in m_list:
        torch.manual_seed(0)
        a = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
        w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
        xn = torch.randn(m, N, device=DEVICE, dtype=DTYPE)

        ref = F.linear(a, w)
        out = v6_gemm(a, w, xn)
        md = (out.float() - ref.float()).abs().max().item()
        assert torch.allclose(out.float(), ref.float(), atol=0.6, rtol=0.02), (
            m,
            md,
        )

        cfgs = {
            "nvjet": lambda: F.linear(a, w),
            "v5_gemm": lambda: v5_gemm(a, w, xn),
            "v6_gemm": lambda: v6_gemm(a, w, xn),
        }
        line = f"M={m:4d} md={md:.3f}"
        for tag, fn in cfgs.items():
            iters_k = bc.measure_graph(fn, flush_buf)
            summ, _ = bc.summarize(iters_k)
            line += f"  {tag}={summ['_total_us']:6.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
