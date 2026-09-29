# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark: fused UP kernel v5 (tcgen05) vs unfused chains at large M.

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/bench_fused_up_v5_largem.py

Per M: fused v5 (tanh + exact sigmoid modes); unfused chains
cuBLAS+hc_gate_mix and cublasLt+hc_gate_mix; cuBLAS GEMM-only; hc_gate_mix
alone (precomputed gate). Same CUPTI / CUDA-graph / cold-L2 methodology as
bench_fused_up_v5.py.
"""

from __future__ import annotations

import json
import statistics
import subprocess

import torch
import torch.nn.functional as F
from flashinfer.gemm import mm_bf16
from flashinfer.testing import bench_gpu_time_with_cupti

from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
from workspace.qwen4_exp_mhc_llgemm.fused_up.up_gate_mix import UpGateMixV5

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD, K, N = 4, 2560, 320, 10240
WARMUP = 25
M_TIMING = [64, 96, 128, 160, 192, 224, 256, 384, 512]

_v5_tanh = UpGateMixV5(sigmoid_mode="tanh")
_v5_exact = UpGateMixV5(sigmoid_mode="exact")


def _bench_us(fn, input_args) -> float:
    for _ in range(WARMUP):
        fn(*input_args)
    torch.cuda.synchronize()
    samples_ms = bench_gpu_time_with_cupti(
        fn,
        use_cuda_graph=True,
        cold_l2_cache=True,
        dry_run_time_ms=25,
        repeat_time_ms=100,
        input_args=input_args,
    )
    return statistics.median(samples_ms) * 1e3


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    torch.cuda.set_device(0)
    rows = []
    for m in M_TIMING:
        torch.manual_seed(0)
        a = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
        w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)
        xn = torch.randn(m, N, device=DEVICE, dtype=DTYPE)
        row = dict(M=m)

        # --- correctness before timing --------------------------------------
        out = _v5_tanh(a, w, xn)
        g = (a.float() @ w.float().T).to(DTYPE).float().view(m, HC, HD)
        ref1 = (torch.sigmoid(g) * xn.float().view(m, HC, HD)).mean(1)
        row["maxdiff_fp32ref"] = (out.float() - ref1).abs().max().item()
        assert torch.allclose(out.float(), ref1, atol=2e-2, rtol=2e-2), m
        ref2 = hc_gate_mix(xn, F.linear(a, w), HC)
        row["maxdiff_prodchain"] = (out.float() - ref2.float()).abs().max().item()
        out_e = _v5_exact(a, w, xn)
        row["maxdiff_prodchain_exact"] = (
            (out_e.float() - ref2.float()).abs().max().item()
        )

        # --- timing ----------------------------------------------------------
        row["fused_v5_us"] = _bench_us(lambda a, w, x: _v5_tanh(a, w, x), (a, w, xn))
        row["fused_v5_exact_us"] = _bench_us(
            lambda a, w, x: _v5_exact(a, w, x), (a, w, xn)
        )
        row["gemm_cublas_us"] = _bench_us(lambda a, w: F.linear(a, w), (a, w))
        gate = F.linear(a, w)
        row["mix_only_us"] = _bench_us(lambda x, g: hc_gate_mix(x, g, HC), (xn, gate))
        row["chain_cublas_us"] = _bench_us(
            lambda a, w, x: hc_gate_mix(x, F.linear(a, w), HC), (a, w, xn)
        )
        row["chain_cublaslt_us"] = _bench_us(
            lambda a, w, x: hc_gate_mix(x, mm_bf16(a, w.t(), backend="cublaslt"), HC),
            (a, w, xn),
        )
        best_key = min(("chain_cublas_us", "chain_cublaslt_us"), key=lambda k: row[k])
        row["best_unfused"] = best_key.removeprefix("chain_").removesuffix("_us")
        row["best_unfused_us"] = row[best_key]
        row["speedup_vs_best"] = row[best_key] / row["fused_v5_us"]
        row["speedup_exact_vs_best"] = row[best_key] / row["fused_v5_exact_us"]
        rows.append(row)
        print(row, flush=True)

    meta = dict(
        gpu=torch.cuda.get_device_name(0),
        dtype="bf16",
        commit=_git_commit(),
        torch=torch.__version__,
    )
    print(json.dumps(meta, indent=2))
    with open(
        "workspace/qwen4_exp_mhc_llgemm/results_fused_up_v5_largem.json", "w"
    ) as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)


if __name__ == "__main__":
    main()
