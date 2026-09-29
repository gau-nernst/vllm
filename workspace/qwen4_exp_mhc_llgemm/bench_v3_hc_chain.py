# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HC down+up chain bench for the v3 FMA candidate (small-M).

Companion to bench_v5_hc_chain.py (which only measured v5 in the chain).
Measures, per M, in one CUDA graph with per-kernel CUPTI timing + cold L2:

  chainB         [down -> F.linear -> hc_gate_mix]   (production up)
  chainA_v5_pdl  [down -> v5 tcgen05 (PDL on)]       (same-run reference)
  chainA_v3_pdl  [down -> v3 FMA (strided lora, PDL on)]

v3 takes the strided lora slice (row stride 336) directly: the compiled
layout keeps the a row stride dynamic, only the wrapper validation used to
require contiguity.

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/bench_v3_hc_chain.py [M_LIST]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import torch
import torch.nn.functional as F

from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import (
    MAX_FUSED_M,
    hc_down_silu,
)
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu
from workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain import (
    measure_graph,
    summarize,
)
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import (
    UpGateMix,
    UpGateMixV5,
)

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD, K, N = 4, 2560, 320, 10240
RANK = 320
DOWN_ROWS = 336
M_LIST = [1, 2, 4, 8, 16]


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST

    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    v5_pdl = UpGateMixV5(sigmoid_mode="tanh", use_pdl=True)
    # The small-M v3 config (max_m=32 variant covers all M measured here).
    v3 = UpGateMix(
        impl="v3",
        max_m=32,
        channels_per_warp=2,
        m_block=4,
        m_splits=4,
        lanes_per_dot=16,
    )

    def down_and_inject(xn):
        m = xn.shape[0]
        if m <= MAX_FUSED_M:
            lora, _ = hc_down_silu(xn, down_w, RANK, HC)
            return lora
        dai = F.linear(xn, down_w)
        return hc_silu(dai[:, :RANK], HC)

    rows = []
    for m in m_list:
        torch.manual_seed(0)
        xn = torch.randn(m, HC * HD, device=DEVICE, dtype=DTYPE)
        down_w = torch.randn(DOWN_ROWS, HC * HD, device=DEVICE, dtype=DTYPE)
        up_w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)

        row = dict(M=m)

        # --- correctness: v3 chain vs production chain -----------------------
        lora = down_and_inject(xn)
        out = v3(lora, up_w, xn)
        ref = hc_gate_mix(xn, F.linear(lora.float().to(DTYPE), up_w), HC)
        row["maxdiff_prodchain"] = (out.float() - ref.float()).abs().max().item()
        assert row["maxdiff_prodchain"] < 2e-2, row["maxdiff_prodchain"]

        # --- measurements ------------------------------------------------------
        def make_chain(up_fn):
            def chain():
                lora_ = down_and_inject(xn)
                up_fn(lora_)

            return chain

        cfgs = {
            "chainB": make_chain(lambda l: hc_gate_mix(xn, F.linear(l, up_w), HC)),
            "chainA_v5_pdl": make_chain(lambda l: v5_pdl(l, up_w, xn)),
            "chainA_v3_pdl": make_chain(lambda l: v3(l, up_w, xn)),
        }
        for tag, fn in cfgs.items():
            iters_k = measure_graph(fn, flush_buf)
            summ, n_it = summarize(iters_k)
            row[tag] = summ
            print(
                f"M={m:3d} {tag:15s} n_it={n_it} "
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
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        torch=torch.__version__,
    )
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_v3_hc_chain.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
