# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Probe: v7 with the xn TMA issued BEFORE the PDL wait (xn_pre_wait).

xn is the chain input (it predates the down kernel), so in the 2-kernel
bench chain it can stream with the weights instead of after the wait.
Production safety needs the down kernel to fire launch_dependents only after
its own griddepcontrol_wait (see TRACKING.md); this probe just measures the
ceiling.

Candidates: v7_early (current) vs v7_xnpre (xn_pre_wait=True), M=8..64.

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/probe_v7_xn_pre.py
"""

from __future__ import annotations

import json
import os
import subprocess

import torch
import torch.nn.functional as F

import workspace.qwen4_exp_mhc_llgemm.bench_hc_chain_pdl as bp  # noqa: F401  (patches bc.classify)
import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
from workspace.qwen4_exp_mhc_llgemm.fused_down.early_pdl.hc_down_silu_early import (
    hc_down_silu_early,
)
from workspace.qwen4_exp_mhc_llgemm.fused_up._up_gemm_v7 import UpGemmV7

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD, K, N = 4, 2560, 320, 10240
RANK = 320
DOWN_ROWS = 336
M_LIST = [8, 16, 32, 48, 64]


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    v7_pdl = UpGemmV7(use_pdl=True, gemm_only=False)
    v7_xnpre = UpGemmV7(use_pdl=True, gemm_only=False, xn_pre_wait=True)

    rows = []
    for m in M_LIST:
        torch.manual_seed(0)
        xn = torch.randn(m, HC * HD, device=DEVICE, dtype=DTYPE)
        down_w = torch.randn(DOWN_ROWS, HC * HD, device=DEVICE, dtype=DTYPE)
        up_w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)

        def down_early(xn):
            lora, _ = hc_down_silu_early(xn, down_w, RANK, HC)
            return lora

        lora_e = down_early(xn)
        ref = hc_gate_mix(xn, F.linear(lora_e.float().to(DTYPE), up_w), HC)

        row = dict(M=m)
        for tag, up in (("v7_early", v7_pdl), ("v7_xnpre", v7_xnpre)):
            md = (up(lora_e, up_w, xn).float() - ref.float()).abs().max().item()
            row[f"maxdiff_{tag}"] = md
            assert md < 2e-2, (m, tag, md)

        def chain_of(up):
            def chain(xn=xn):
                up(down_early(xn), up_w, xn)

            return chain

        for tag, up in (("v7_early", v7_pdl), ("v7_xnpre", v7_xnpre)):
            iters_k = bc.measure_graph(chain_of(up), flush_buf)
            summ, n_it = bc.summarize(iters_k)
            row[tag] = summ
            print(
                f"M={m:3d} {tag:10s} n_it={n_it} "
                + " ".join(
                    f"{k}={v['dur_us']:.2f}@{v['start_us']:.2f}"
                    for k, v in summ.items()
                    if k != "_total_us"
                )
                + f" total={summ['_total_us']:.2f}",
                flush=True,
            )
        rows.append(row)

    meta = dict(
        gpu=torch.cuda.get_device_name(0),
        dtype="bf16",
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        torch=torch.__version__,
    )
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_v7_xn_pre.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
