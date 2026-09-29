# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Production-shaped HC down+up chain bench: PDL init-hiding for v5.

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/bench_v5_hc_chain.py

Chain (production mix() without the rmsnorm):
    hc_down_silu -> up
- down = merged production op (fused cute-dsl kernel for M<=48,
  F.linear+hc_silu for M>48; it self-compiles on first eager call).
- The dependency edge is real: lora = down_out[:, :320] (strided, row
  stride 336) feeds v5's activation input for M<=48; for M>48 hc_silu
  returns a contiguous [M, 320] tensor, like production.

Per M:
  standalone_nopdl    v5 alone (contiguous a), PDL off -- reference
  standalone_strided  v5 alone (strided a, stride 336), PDL off
  chainA_pdl          [down -> v5(PDL on)]
  chainA_nopdl        [down -> v5(PDL off)]
  chainB              [down -> F.linear -> hc_gate_mix]  (production up)

Per-kernel CUPTI times inside each CUDA graph + chain span. Cold L2 via a
2x-L2 flush kernel between iterations (same mechanism as
flashinfer's bench_gpu_time_with_cupti), CUDA graph, 25 warmup.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from functools import partial

import torch
import torch.nn.functional as F

from vllm.models.qwen4_exp.nvidia.ops.cute_dsl.hc_down_silu import (
    MAX_FUSED_M,
    hc_down_silu,
)
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix, hc_silu
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import UpGateMixV5

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD, K, N = 4, 2560, 320, 10240
RANK = 320
DOWN_ROWS = 336  # rank + hc + pad (production merged down+inject weight)
M_LIST = [1, 2, 4, 8, 16, 32, 48, 64, 96]
WARMUP = 25
ITERS = 200


# ---------------------------------------------------------------- CUPTI ----
def measure_graph(fn, flush_buf, iters=ITERS):
    """Capture fn in a CUDA graph; return per-iteration kernel records.

    Returns a list of iterations, each a list of (name, start_ns, end_ns)
    for CONCURRENT_KERNEL activities (flush fill excluded by segmentation).
    """
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(5):
        flush_buf.zero_()
        g.replay()
    torch.cuda.synchronize()

    from cupti import cupti

    records = []

    def buf_requested():
        return 8 * 1024 * 1024, 0

    def buf_completed(recs, activities):
        for a in activities:
            if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL:
                recs.append((a.name, a.start, a.end))

    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_register_callbacks(buf_requested, partial(buf_completed, records))
    for _ in range(iters):
        flush_buf.zero_()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.finalize()

    records.sort(key=lambda r: r[1])

    # Segment iterations at the L2-flush fill kernel (name-based: the
    # flush is a FillFunctor on the 2x-L2 int8 buffer).
    def is_flush(n: str) -> bool:
        return "Fill" in n or "fill" in n or "Memset" in n

    iters_out: list[list[tuple[str, int, int]]] = []
    cur: list[tuple[str, int, int]] = []
    for name, st, en in records:
        if is_flush(name):
            if cur:
                iters_out.append(cur)
                cur = []
        else:
            cur.append((name, st, en))
    if cur:
        iters_out.append(cur)
    return iters_out


def classify(name: str) -> str:
    if "UpGateMixV5" in name or "up_gate_mix_v5" in name:
        return "v5"
    if "_hc_gate_mix" in name:
        return "gate_mix"
    if "_hc_silu" in name:
        return "silu"
    if "HcDownSilu" in name or "hc_down_silu" in name:
        return "down"
    if "splitKreduce" in name:
        return "splitk_reduce"
    if "nvjet" in name.lower() or "gemm" in name.lower() or "cutlass" in name.lower():
        return "cublas"
    return "other:" + name[:60]


def summarize(iters_kernels):
    """Per kernel position: median duration and median start offset from the
    iteration's first kernel start; plus the iteration span. Position keys
    (``f"{pos}:{class}"``) distinguish same-class kernels (e.g. the two
    cuBLAS GEMMs in chainB)."""
    seq0 = tuple(classify(n) for n, _, _ in iters_kernels[0])
    for it in iters_kernels:
        if tuple(classify(n) for n, _, _ in it) != seq0:
            raise ValueError(f"inconsistent iteration: {[n for n, _, _ in it]}")
    out = {}
    for pos, (n, _, _) in enumerate(iters_kernels[0]):
        key = f"{pos}:{classify(n)}"
        durs, starts = [], []
        for it in iters_kernels:
            t0 = it[0][1]
            _, s, e = it[pos]
            durs.append(e - s)
            starts.append(s - t0)
        out[key] = dict(
            dur_us=statistics.median(durs) / 1e3,
            start_us=statistics.median(starts) / 1e3,
        )
    spans = [
        max(e for _, _, e in it) - min(s for _, s, _ in it) for it in iters_kernels
    ]
    out["_total_us"] = statistics.median(spans) / 1e3
    return out, len(iters_kernels)


# ------------------------------------------------------------------ chain ----
def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST

    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    v5_pdl = UpGateMixV5(sigmoid_mode="tanh", use_pdl=True)
    v5_nopdl = UpGateMixV5(sigmoid_mode="tanh", use_pdl=False)

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
        a_contig = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
        a_strided = torch.randn(m, DOWN_ROWS, device=DEVICE, dtype=DTYPE)[:, :K]

        row = dict(M=m)

        # --- correctness of the strided-a path -------------------------------
        lora = down_and_inject(xn)
        assert lora.shape == (m, K)
        out = v5_pdl(lora, up_w, xn)
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
            "standalone_nopdl": lambda: v5_nopdl(a_contig, up_w, xn),
            "standalone_strided": lambda: v5_nopdl(a_strided, up_w, xn),
            "chainA_pdl": make_chain(lambda l: v5_pdl(l, up_w, xn)),
            "chainA_nopdl": make_chain(lambda l: v5_nopdl(l, up_w, xn)),
            "chainB": make_chain(lambda l: hc_gate_mix(xn, F.linear(l, up_w), HC)),
        }
        for tag, fn in cfgs.items():
            iters_k = measure_graph(fn, flush_buf)
            summ, n_it = summarize(iters_k)
            row[tag] = summ
            print(
                f"M={m:3d} {tag:19s} n_it={n_it} "
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
        os.path.dirname(os.path.abspath(__file__)), "results_v5_hc_chain.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
