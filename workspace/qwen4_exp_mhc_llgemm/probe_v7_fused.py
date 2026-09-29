# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v7 fused gate-mix: correctness vs torch reference + v6, bench vs v5/v6.

PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/probe_v7_fused.py [M_LIST]
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from workspace.qwen4_exp_mhc_llgemm.fused_up._up_gemm_v7 import UpGemmV7
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import (
    UpGateMixV5,
    UpGateMixV6,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
K, N, HC = 320, 10240, 4
HD = N // HC
M_LIST = [1, 8, 16, 32, 48, 64, 96, 128]


def ref_gate_mix(a, w, xn):
    # Production semantics: fp32 GEMM -> bf16 -> fp32 -> sigmoid; s-ordered
    # serial mix; scale 1/HC; bf16 out.
    acc = F.linear(a, w).float().view(-1, HC, HD)
    x = xn.float().view(-1, HC, HD)
    sig = torch.sigmoid(acc.to(DTYPE).float())
    out = (sig[:, 0] * x[:, 0] + sig[:, 1] * x[:, 1]) + sig[:, 2] * x[:, 2]
    out = out + sig[:, 3] * x[:, 3]
    return (out * (1.0 / HC)).to(DTYPE)


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST
    flush_buf = torch.empty(
        2 * get_l2_cache_size(DEVICE), device=DEVICE, dtype=torch.int8
    )

    v5 = UpGateMixV5(use_pdl=False)
    v6 = UpGateMixV6(use_pdl=False)
    v7 = UpGemmV7(use_pdl=False, gemm_only=False)

    for m in m_list:
        torch.manual_seed(0)
        a = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
        w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
        xn = torch.randn(m, N, device=DEVICE, dtype=DTYPE)

        ref = ref_gate_mix(a, w, xn)
        out6 = v6(a, w, xn)
        out7 = v7(a, w, xn)
        md_ref = (out7.float() - ref.float()).abs().max().item()
        md_v6 = (out7.float() - out6.float()).abs().max().item()
        assert torch.allclose(out7.float(), ref.float(), atol=0.05, rtol=0.02), (
            m,
            md_ref,
        )

        cfgs = {
            "v5_fused": lambda: v5(a, w, xn),
            "v6_fused": lambda: v6(a, w, xn),
            "v7_fused": lambda: v7(a, w, xn),
        }
        line = f"M={m:4d} md_ref={md_ref:.4f} md_v6={md_v6:.4f}"
        for tag, fn in cfgs.items():
            iters_k = bc.measure_graph(fn, flush_buf)
            summ, _ = bc.summarize(iters_k)
            line += f"  {tag}={summ['_total_us']:6.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
