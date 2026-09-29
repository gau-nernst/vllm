# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""3-kernel mHC chain bench: [combine_norm -> down -> up].

Production's hc_combine_norm fires gdc_launch_dependents ~60% into the
kernel (after the combine stores, before the norm tail). The early variant
fires at CTA start, so the PDL-launched down kernel can pre-stream its
weights during ALL of combine_norm's body (combine_norm's grid is M*HC CTAs
-- no SM-capacity conflict with down's grid).

Columns:
  cnprod     prod combine_norm  -> down(early) -> up (v3 M<=4, v7 8-48)
  cnearly    early combine_norm -> down(early) -> up
  cnearly_pf (M<=4 only) cnearly + down FMA prefetch_pdl_weights=True
             (extends the pre-wait weight stream beyond M=1)

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/bench_hc_chain3.py [M_LIST]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import torch

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_combine_norm
from workspace.qwen4_exp_mhc_llgemm.fused_down.early_pdl.hc_combine_norm_early import (
    hc_combine_norm_early,
)
from workspace.qwen4_exp_mhc_llgemm.fused_down.early_pdl.hc_down_silu_early import (
    _get_kernel_early,
    hc_down_silu_early,
)
from workspace.qwen4_exp_mhc_llgemm.fused_up._up_gemm_v7 import UpGemmV7
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import UpGateMix

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD, K, N = 4, 2560, 320, 10240
RANK = 320
DOWN_ROWS = 336
EPS = 1e-6
M_LIST = [1, 2, 4, 8, 16, 32, 48]


def classify(name: str) -> str:
    if "UpGemmV7" in name or "up_gemm_v7" in name:
        return "v7"
    if "combine_norm_early" in name:
        return "cn_early"
    if "combine_norm" in name:
        return "cn"
    if "HcDownSilu" in name or "hc_down_silu" in name:
        return "down"
    if "cutlass" in name.lower() or "gemm" in name.lower():
        return "v3"
    return "other:" + name[:60]


bc.classify = classify


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST

    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    v7_pdl = UpGemmV7(use_pdl=True, gemm_only=False)
    v3 = UpGateMix(
        impl="v3",
        max_m=32,
        channels_per_warp=2,
        m_block=4,
        m_splits=4,
        lanes_per_dot=16,
    )

    rows = []
    for m in m_list:
        torch.manual_seed(0)
        hidden = torch.randn(m, HC * HD, device=DEVICE, dtype=DTYPE)
        block = torch.randn(m, HD, device=DEVICE, dtype=DTYPE)
        inj = torch.randn(m, HC, device=DEVICE, dtype=DTYPE)
        norm_w = torch.randn(HC * HD, device=DEVICE, dtype=DTYPE)
        down_w = torch.randn(DOWN_ROWS, HC * HD, device=DEVICE, dtype=DTYPE)
        up_w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)

        up = (
            (lambda l, xn: v3(l, up_w, xn))
            if m <= 4
            else (lambda l, xn: v7_pdl(l, up_w, xn))
        )

        # --- correctness: early combine_norm must be bit-identical ----------
        out_p, xn_p = hc_combine_norm(hidden, block, inj, norm_w, EPS, HC)
        out_e, xn_e = hc_combine_norm_early(hidden, block, inj, norm_w, EPS, HC)
        row = dict(M=m)
        row["cn_early_bitidentical"] = bool(
            torch.equal(out_p, out_e) and torch.equal(xn_p, xn_e)
        )
        assert row["cn_early_bitidentical"]

        def chain(cn_fn, prefetch=False):
            def run():
                _, xn = cn_fn(hidden, block, inj, norm_w, EPS, HC)
                if m <= 4 and prefetch:
                    kern = _get_kernel_early(RANK, HC, HC * HD, True)
                    lora = kern(xn, down_w)[:, :RANK]
                else:
                    lora, _ = hc_down_silu_early(xn, down_w, RANK, HC)
                up(lora, xn)

            return run

        cfgs = {
            "cnprod": chain(hc_combine_norm),
            "cnearly": chain(hc_combine_norm_early),
        }
        if m <= 4:
            cfgs["cnearly_pf"] = chain(hc_combine_norm_early, prefetch=True)

        for tag, fn in cfgs.items():
            iters_k = bc.measure_graph(fn, flush_buf)
            summ, n_it = bc.summarize(iters_k)
            row[tag] = summ
            print(
                f"M={m:3d} {tag:11s} n_it={n_it} "
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
        os.path.dirname(os.path.abspath(__file__)), "results_hc_chain3.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
