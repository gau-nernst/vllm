# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v7 fused-epilogue split sweep: more warpgroups for wide token tiles.

probe_v7_phases.py showed the epilogue costs ~1 us per 32 serial tokens per
thread (~31 ns/token), so wide mma_n tiles are epilogue-bound, not
stream-bound. The fused epilogue already splits tokens across 2 warpgroups at
mma_n>=64; this probe sweeps epi_split (1/2/4/8) at M=32..128.

Correctness: each output token is computed by exactly one thread with the
same op order regardless of the split, so results should be BIT-IDENTICAL to
the default split.

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/probe_v7_episplit.py
"""

from __future__ import annotations

import statistics

import torch

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from workspace.qwen4_exp_mhc_llgemm.fused_up._up_gemm_v7 import UpGemmV7

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, K, N = 4, 320, 10240


def dur_us(fn, flush_buf):
    iters = bc.measure_graph(fn, flush_buf)
    durs = []
    for it in iters:
        assert len(it) == 1, [n for n, _, _ in it]
        _, st, en = it[0]
        durs.append(en - st)
    return statistics.median(durs) / 1e3


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    torch.manual_seed(0)
    w = torch.randn(N, K, device=DEVICE, dtype=DTYPE)

    for m in (32, 48, 64, 96, 128):
        a = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
        xn = torch.randn(m, N, device=DEVICE, dtype=DTYPE)
        mma_n = next(b for b in (8, 16, 32, 64, 96, 128) if m <= b)
        ref = None
        parts = []
        for split in (1, 2, 4, 8):
            if mma_n % split or mma_n // split < 8:
                continue
            up = UpGemmV7(
                gemm_only=False, mma_n_override=mma_n, epi_split_override=split
            )
            out = up(a, w, xn)
            if ref is None:
                ref = out
            else:
                md = (out.float() - ref.float()).abs().max().item()
                assert md == 0.0, (m, split, md)
            t = dur_us(lambda: up(a, w, xn), flush_buf)
            parts.append(f"EPI{split}={t:.2f}")
        print(f"M={m:3d} mma_n={mma_n:3d} " + " ".join(parts), flush=True)


if __name__ == "__main__":
    main()
