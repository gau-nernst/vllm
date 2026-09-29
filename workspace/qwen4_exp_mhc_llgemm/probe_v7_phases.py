# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v7 phase decomposition: where does the time go at mma_n = 32..128?

From probe_stream_floor.py: a fixed-size cold stream is flat in CTA count
past ~80 CTAs, TMA beats LDG, and the 6.25 MB weight stream alone costs
3.90 us. v7 GEMM-only takes 6.78/7.84/8.99 us at M=64/96/128 -- 2-4 us above
the stream line, growing with mma_n. This probe splits the kernel into

  stream     = W+A TMA only (no MMA, no epilogue)      [probe_phase="stream"]
  mma_nowait = all MMAs issued immediately, no waits   [probe_phase="mma_nowait"]
  serial_mma = wait ALL stages, then all MMAs          [probe_phase="serial_mma"]
  no_epi     = stream + pipelined MMA, no epilogue     [probe_phase="no_epi"]
  full       = stream + MMA + plain GEMM epilogue      [probe_phase="full"]

so the MMA throughput, MMA/stream overlap efficiency, commit/drain latency,
and epilogue contributions at each mma_n are isolated. Outputs are garbage
in the probe phases; timing only.

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/probe_v7_phases.py
"""

from __future__ import annotations

import statistics

import torch

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from workspace.qwen4_exp_mhc_llgemm.fused_up._up_gemm_v7 import UpGemmV7

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
K, N = 320, 10240


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

    for mma_n in (32, 64, 96, 128):
        m = mma_n  # full tile: worst case for the epilogue
        a = torch.randn(m, K, device=DEVICE, dtype=DTYPE)
        times = {}
        for phase in (
            "stream",
            "commit_only",
            "mma_nowait",
            "serial_mma",
            "no_fence",
            "late_mma_1",
            "late_mma_4",
        ):
            up = UpGemmV7(gemm_only=True, mma_n_override=mma_n, probe_phase=phase)
            up(a, w)  # compile
            times[phase] = dur_us(lambda: up(a, w), flush_buf)
        for phase in ("no_epi", "full"):
            for spin in (False, True):
                up = UpGemmV7(
                    gemm_only=True,
                    mma_n_override=mma_n,
                    probe_phase=phase,
                    spin_wait=spin,
                )
                up(a, w)
                times[f"{phase}{'_spin' if spin else ''}"] = dur_us(
                    lambda: up(a, w), flush_buf
                )
        print(
            f"mma_n={mma_n:3d} stream={times['stream']:.2f} "
            f"commit_only={times['commit_only']:.2f} "
            f"(+{times['commit_only'] - times['stream']:.2f}) "
            f"mma_nowait={times['mma_nowait']:.2f} "
            f"serial_mma={times['serial_mma']:.2f} "
            f"(mma_thr=+{times['serial_mma'] - times['stream']:.2f}) "
            f"no_fence={times['no_fence']:.2f} "
            f"late1=+{times['late_mma_1'] - times['stream']:.2f} "
            f"late4=+{times['late_mma_4'] - times['stream']:.2f} "
            f"no_epi={times['no_epi']:.2f}/{times['no_epi_spin']:.2f}spin "
            f"full={times['full']:.2f}/{times['full_spin']:.2f}spin",
            flush=True,
        )


if __name__ == "__main__":
    main()
