# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: v6 (standard API, gn-kernels base) vs v5 (experimental API).

Standalone, graph-replay, cold-L2 CUPTI, PDL off. Also checks v6 strided-a
(row stride 336, the production lora slice) correctness.

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/probe_v6.py [M_LIST]
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import (
    UpGateMixV5,
    UpGateMixV6,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, K, N = 4, 320, 10240
M_LIST = [1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256]


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST
    flush_buf = torch.empty(
        2 * get_l2_cache_size(DEVICE), device=DEVICE, dtype=torch.int8
    )

    v5 = UpGateMixV5(use_pdl=False)
    v6 = UpGateMixV6(use_pdl=False)

    for m in m_list:
        torch.manual_seed(0)
        a = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
        w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
        xn = torch.randn(m, N, device=DEVICE, dtype=DTYPE)
        gate = F.linear(a, w)
        ref = hc_gate_mix(xn, gate, HC)

        # strided-a correctness (production lora slice, row stride 336)
        a_str = torch.randn(m, 336, device=DEVICE, dtype=DTYPE)[:, :K]
        gate_str = F.linear(a_str, w)
        ref_str = hc_gate_mix(xn, gate_str, HC)
        out_str = v6(a_str, w, xn)
        md_str = (out_str.float() - ref_str.float()).abs().max().item()
        assert md_str < 2e-2, ("strided", m, md_str)

        out6 = v6(a, w, xn)
        md6 = (out6.float() - ref.float()).abs().max().item()
        assert md6 < 2e-2, ("v6", m, md6)

        line = f"M={m:4d}"
        for tag, k in (("v5", v5), ("v6", v6)):
            iters_k = bc.measure_graph(lambda k=k: k(a, w, xn), flush_buf)
            summ, _ = bc.summarize(iters_k)
            line += f"  {tag}={summ['_total_us']:6.2f}"
        line += f"  (md6={md6:.3f} md_str={md_str:.3f})"
        print(line, flush=True)


if __name__ == "__main__":
    main()
