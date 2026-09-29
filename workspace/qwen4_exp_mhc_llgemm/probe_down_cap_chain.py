# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Phase 3a probe: cp.async-pipelined grid-capped down FMA (M<=4) + PDL up.

The uncapped down FMA kernel fills every SM with 2-3 resident CTAs, so the
PDL-dependent up kernel cannot be placed until down drains (~0.9us before
the end). Plain capped variants (LDG) starve the memory pipeline (~1.2us per
extra column/CTA). The cp.async variant keeps (S-1) columns in flight in
smem per CTA, so a capped grid should still sustain the stream while freeing
SMs for the up kernel.

Candidates per M in {1,2,4}:
  v3_<tag>   down(cp.async J cols/CTA, S stages) -> v3 FMA (PDL)
  v3_early   down_early(uncapped) -> v3 FMA (PDL)  (control)

Also checks capped-down output is bit-identical to production down.

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/probe_down_cap_chain.py
"""

from __future__ import annotations

import json
import os
import subprocess

import torch
import torch.nn.functional as F

import workspace.qwen4_exp_mhc_llgemm.bench_hc_chain_pdl as bp  # noqa: F401  (patches bc.classify)
import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import hc_down_silu
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
from workspace.qwen4_exp_mhc_llgemm.fused_down.early_pdl.hc_down_silu_early import (
    hc_down_silu_early,
)
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import UpGateMix

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD, K, N = 4, 2560, 320, 10240
RANK = 320
DOWN_ROWS = 336
M_LIST = [1, 2, 4]
# (tag, fma_cpasync=(cols_per_cta, num_stages)); grid = ceil(324 / J)
VARIANTS = [
    ("cp_j2s3", (2, 3)),  # 162 CTAs (no free SMs; sanity)
    ("cp_j3s4", (3, 4)),  # 108 CTAs, 40 free SMs
    ("cp_j4s4", (4, 4)),  # 81 CTAs, 67 free SMs
    ("cp_j4s6", (4, 6)),  # deeper pipeline
    ("cp_j6s6", (6, 6)),  # 54 CTAs, 94 free SMs
]


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    v3 = UpGateMix(
        impl="v3",
        max_m=32,
        channels_per_warp=2,
        m_block=4,
        m_splits=4,
        lanes_per_dot=16,
    )

    rows = []
    for m in M_LIST:
        torch.manual_seed(0)
        xn = torch.randn(m, HC * HD, device=DEVICE, dtype=DTYPE)
        down_w = torch.randn(DOWN_ROWS, HC * HD, device=DEVICE, dtype=DTYPE)
        up_w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)

        lora_p, _ = hc_down_silu(xn, down_w, RANK, HC)
        ref = hc_gate_mix(xn, F.linear(lora_p.float().to(DTYPE), up_w), HC)

        row = dict(M=m)
        timed = {}

        def down_early(xn):
            lora, _ = hc_down_silu_early(xn, down_w, RANK, HC)
            return lora

        def chain_early(xn=xn):
            v3(down_early(xn), up_w, xn)

        timed["v3_early"] = chain_early

        for tag, js in VARIANTS:

            def chain(xn=xn, js=js):
                lora, _ = hc_down_silu_early(xn, down_w, RANK, HC, fma_cpasync=js)
                v3(lora, up_w, xn)

            lora_c, _ = hc_down_silu_early(xn, down_w, RANK, HC, fma_cpasync=js)
            bit = bool(torch.equal(lora_c.contiguous(), lora_p.contiguous()))
            row[f"{tag}_bitidentical"] = bit
            md = (v3(lora_c, up_w, xn).float() - ref.float()).abs().max().item()
            row[f"{tag}_maxdiff_v3"] = md
            assert bit and md < 2e-2, (m, tag, bit, md)
            timed[f"v3_{tag}"] = chain

        for tag, fn in timed.items():
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
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip(),
        torch=torch.__version__,
        variants=[(t, list(js)) for t, js in VARIANTS],
    )
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results_down_cpasync_chain.json",
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
