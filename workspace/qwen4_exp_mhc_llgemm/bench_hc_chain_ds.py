# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Downstream-overlap bench: [down -> up -> downstream GEMM].

Question: how much of the up kernel's tail (tcgen05 drain + gate-mix
epilogue + TMA store) is hidden by the downstream kernel's startup, and how
much of the downstream's startup hides under our mainloop (v7 fires
launch_dependents right after issuing its TMAs)?

The downstream stands in for the block that consumes mHC `block_input`
(attention qkv / MLP gate_up): F.linear over [M, 2560] -> [M, DS_N].

Candidates:
  base       [down(early PDL) -> v7(PDL)]               (no downstream)
  base_ds    [down(early) -> v7(PDL) -> F.linear]
  v3_ds      [down(early) -> v3 FMA(PDL) -> F.linear]   (M <= 16)
  v5_ds      [down(early) -> v5(PDL) -> F.linear]
  prod_ds    [down(early) -> F.linear -> hc_gate_mix -> F.linear]

The CUPTI start timestamps show whether the downstream launches before the
up kernel finishes (PDL edge) and how much startup it hides.

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/bench_hc_chain_ds.py [M_LIST]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import (
    MAX_FUSED_M,
)
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu
from workspace.qwen4_exp_mhc_llgemm.fused_down.early_pdl.hc_down_silu_early import (
    hc_down_silu_early,
)
from workspace.qwen4_exp_mhc_llgemm.fused_up._up_gemm_v7 import UpGemmV7
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import (
    UpGateMix,
    UpGateMixV5,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD, K, N = 4, 2560, 320, 10240
RANK = 320
DOWN_ROWS = 336
DS_N = 2048  # downstream GEMM rows ([DS_N, 2560] weights = 10 MB)
M_LIST = [1, 2, 4, 8, 16, 32, 48, 64]


def classify(name: str) -> str:
    if "UpGemmV7" in name or "up_gemm_v7" in name:
        return "v7"
    if "UpGateMixV5" in name or "up_gate_mix_v5" in name:
        return "v5"
    if "UpGateMixV3" in name or "up_gate_mix_v3" in name:
        return "v3"
    if "HcDownSilu" in name or "hc_down_silu" in name:
        return "down_early" if "Early" in name else "down"
    if "_hc_gate_mix" in name:
        return "gate_mix"
    if "_hc_silu" in name:
        return "silu"
    if "splitKreduce" in name:
        return "splitk_reduce"
    if "nvjet" in name.lower():
        return "cublas"
    return "other:" + name[:60]


bc.classify = classify  # summarize() resolves classify from its module globals


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST

    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    v7_pdl = UpGemmV7(use_pdl=True, gemm_only=False)
    v5_pdl = UpGateMixV5(sigmoid_mode="tanh", use_pdl=True)
    # Small-M v3 config (same as bench_v3_hc_chain.py); covers M <= 16.
    v3 = UpGateMix(
        impl="v3",
        max_m=32,
        channels_per_warp=2,
        m_block=4,
        m_splits=4,
        lanes_per_dot=16,
    )

    def down_early(xn):
        m = xn.shape[0]
        if m <= MAX_FUSED_M:
            lora, _ = hc_down_silu_early(xn, down_w, RANK, HC)
            return lora
        dai = F.linear(xn, down_w)
        return hc_silu(dai[:, :RANK], HC)

    rows = []
    for m in m_list:
        torch.manual_seed(0)
        xn = torch.randn(m, HC * HD, device=DEVICE, dtype=DTYPE)
        down_w = torch.randn(DOWN_ROWS, HC * HD, device=DEVICE, dtype=DTYPE)
        up_w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
        ds_w = torch.randn(DS_N, HD, device=DEVICE, dtype=DTYPE)

        row = dict(M=m)

        # --- measurements ------------------------------------------------------
        def base():
            lora_ = down_early(xn)
            v7_pdl(lora_, up_w, xn)

        def base_ds():
            lora_ = down_early(xn)
            blk = v7_pdl(lora_, up_w, xn)
            F.linear(blk, ds_w)

        def v3_ds():
            lora_ = down_early(xn)
            blk = v3(lora_, up_w, xn)
            F.linear(blk, ds_w)

        def v5_ds():
            lora_ = down_early(xn)
            blk = v5_pdl(lora_, up_w, xn)
            F.linear(blk, ds_w)

        def prod_ds():
            lora_ = down_early(xn)
            blk = hc_gate_mix(xn, F.linear(lora_, up_w), HC)
            F.linear(blk, ds_w)

        cands = [
            ("base", base),
            ("prod_ds", prod_ds),
            ("v5_ds", v5_ds),
            ("base_ds", base_ds),
        ]
        if m <= 16:
            cands.append(("v3_ds", v3_ds))
        for tag, fn in cands:
            iters_k = bc.measure_graph(fn, flush_buf)
            summ, n_it = bc.summarize(iters_k)
            row[tag] = summ
            print(
                f"M={m:3d} {tag:8s} n_it={n_it} "
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
        os.path.dirname(os.path.abspath(__file__)), "results_hc_chain_ds.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
