# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HC down+up chain bench: early-launch_dependents PDL round.

Four candidates, all in one CUDA graph with per-kernel CUPTI timing + cold L2:

  prod          [down(prod) -> F.linear -> hc_gate_mix]   (original baseline)
  unfused_early [down(early launch_dependents) -> F.linear -> hc_gate_mix]
                (control: does the down-PDL change help a non-PDL dependent?)
  v3_early      [down(early) -> v3 FMA (PDL)]
  v5_early      [down(early) -> v5 tcgen05 (PDL)]
  v7_early      [down(early) -> v7 tcgen05 MMA_M=128 (PDL)]
  v7_nopdl      [down(early) -> v7, no PDL]   (no overlap possible: floor)
  v7_waitfirst  [down(early) -> v7, PDL wait before weight TMA]
                (no weight pre-streaming: isolates the pre-stream overlap)

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/bench_hc_chain_pdl.py [M_LIST]
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
    hc_down_silu,
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
M_LIST = [1, 2, 4, 8, 16, 32, 48, 64, 96]
V3_MAX_M = 8  # v3 only where it can win (standalone crossover is at 8)


def classify(name: str) -> str:
    if "UpGemmV7" in name or "up_gemm_v7" in name:
        return "v7"
    if "UpGateMixV5" in name or "up_gate_mix_v5" in name:
        return "v5"
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
    if "cutlass" in name.lower() or "gemm" in name.lower():
        return "v3"
    return "other:" + name[:60]


bc.classify = classify  # summarize() resolves classify from its module globals


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST

    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    v5_pdl = UpGateMixV5(sigmoid_mode="tanh", use_pdl=True)
    v7_pdl = UpGemmV7(use_pdl=True, gemm_only=False)
    # Ordering probes for the PDL pre-stream hypothesis:
    # v7_nopdl: no PDL at all -> no overlap possible (chain floor).
    # v7_waitfirst: PDL wait BEFORE the weight stream -> no pre-streaming.
    v7_nopdl = UpGemmV7(use_pdl=False, gemm_only=False)
    v7_waitfirst = UpGemmV7(use_pdl=True, gemm_only=False, pdl_wait_first=True)
    v3 = UpGateMix(
        impl="v3",
        max_m=32,
        channels_per_warp=2,
        m_block=4,
        m_splits=4,
        lanes_per_dot=16,
    )

    def down_prod(xn):
        m = xn.shape[0]
        if m <= MAX_FUSED_M:
            lora, _ = hc_down_silu(xn, down_w, RANK, HC)
            return lora
        dai = F.linear(xn, down_w)
        return hc_silu(dai[:, :RANK], HC)

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

        row = dict(M=m)

        # --- correctness -------------------------------------------------------
        lora_p = down_prod(xn)
        lora_e = down_early(xn)
        row["down_early_bitidentical"] = bool(
            torch.equal(lora_p.contiguous(), lora_e.contiguous())
        )
        ref = hc_gate_mix(xn, F.linear(lora_p.float().to(DTYPE), up_w), HC)
        if m <= V3_MAX_M:
            row["maxdiff_v3"] = (
                (v3(lora_e, up_w, xn).float() - ref.float()).abs().max().item()
            )
            assert row["maxdiff_v3"] < 2e-2, row["maxdiff_v3"]
        row["maxdiff_v5"] = (
            (v5_pdl(lora_e, up_w, xn).float() - ref.float()).abs().max().item()
        )
        assert row["maxdiff_v5"] < 2e-2, row["maxdiff_v5"]
        row["maxdiff_v7"] = (
            (v7_pdl(lora_e, up_w, xn).float() - ref.float()).abs().max().item()
        )
        assert row["maxdiff_v7"] < 2e-2, row["maxdiff_v7"]
        for tag, up in (("v7_nopdl", v7_nopdl), ("v7_waitfirst", v7_waitfirst)):
            md = (up(lora_e, up_w, xn).float() - ref.float()).abs().max().item()
            row[f"maxdiff_{tag}"] = md
            assert md < 2e-2, (tag, md)

        # --- measurements ------------------------------------------------------
        def make_chain(down_fn, up_fn):
            def chain():
                lora_ = down_fn(xn)
                up_fn(lora_)

            return chain

        unfused_up = lambda l: hc_gate_mix(xn, F.linear(l, up_w), HC)
        cfgs = {
            "prod": make_chain(down_prod, unfused_up),
            "unfused_early": make_chain(down_early, unfused_up),
            "v5_early": make_chain(down_early, lambda l: v5_pdl(l, up_w, xn)),
            "v7_early": make_chain(down_early, lambda l: v7_pdl(l, up_w, xn)),
            "v7_nopdl": make_chain(down_early, lambda l: v7_nopdl(l, up_w, xn)),
            "v7_waitfirst": make_chain(down_early, lambda l: v7_waitfirst(l, up_w, xn)),
        }
        if m <= V3_MAX_M:
            cfgs["v3_early"] = make_chain(down_early, lambda l: v3(l, up_w, xn))
        for tag, fn in cfgs.items():
            iters_k = bc.measure_graph(fn, flush_buf)
            summ, n_it = bc.summarize(iters_k)
            row[tag] = summ
            print(
                f"M={m:3d} {tag:14s} n_it={n_it} "
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
        os.path.dirname(os.path.abspath(__file__)), "results_hc_chain_pdl.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
