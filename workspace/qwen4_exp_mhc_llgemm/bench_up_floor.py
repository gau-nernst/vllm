# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone floor bench for the fused up kernel: baseline components.

Per M, each captured alone in a CUDA graph with cold-L2 CUPTI timing:

  gemm      F.linear(a[M,320], up_w[10240,320])  (cuBLAS nvjet)
  gate_mix  hc_gate_mix(xn, gate)                (triton, PDL on)

gemm alone is the optimization floor for the fused kernel; gemm + gate_mix
is the unfused baseline.

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/bench_up_floor.py [M_LIST]
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.nn.functional as F

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD, K, N = 4, 2560, 320, 10240
M_LIST = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 384, 512]


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST
    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    rows = []
    for m in m_list:
        torch.manual_seed(0)
        a = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
        up_w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
        xn = torch.randn(m, N, device=DEVICE, dtype=DTYPE)
        gate = torch.randn(m, N, device=DEVICE, dtype=DTYPE)

        row = dict(M=m)
        for tag, fn in {
            "gemm": lambda: F.linear(a, up_w),
            "gate_mix": lambda: hc_gate_mix(xn, gate, HC),
        }.items():
            iters_k = bc.measure_graph(fn, flush_buf)
            summ, n_it = bc.summarize(iters_k)
            row[tag] = summ["_total_us"]
            print(
                f"M={m:3d} {tag:8s} n_it={n_it} total={summ['_total_us']:.2f}",
                flush=True,
            )
        rows.append(row)

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_up_floor.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
