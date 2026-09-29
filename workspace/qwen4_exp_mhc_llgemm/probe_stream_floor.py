# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw cold-stream floor probe: what limits the mHC up GEMM's weight stream?

The v7 up GEMM streams 6.5 MB of read-once weights in ~4.4-4.7 us cold
(~1.4 TB/s aggregate over 80 CTAs), far below what HBM should do. This probe
streams a read-once buffer with v7-style fire-and-forget TMA boxes (and an
LDG.128 baseline) and sweeps:

  - CTA count G (concurrency): 40 .. 296 (2/SM)
  - per-CTA bytes (32-160 KB)
  - TMA in-flight depth: stages x box shape at fixed total bytes
  - cold L2 (flush between iters) vs warm L2

so we can tell whether the floor is CTA-concurrency-limited (fix: more CTAs
per weight stream, e.g. BLOCK_N=80/96 v7 tilings), in-flight-depth-limited
(fix: deeper/narrower boxes), or a hard HBM/L2 wall (no kernel fix).

Run from the repository root:

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/probe_stream_floor.py
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.cute_utils import simple_tma_copy

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
COLS = 64  # bf16 per row = 128B
BUF_BYTES = 64 * 1024 * 1024
G_LIST = [40, 80, 107, 128, 148, 160, 222, 296]


class StreamTMA:
    """Fire-and-forget TMA stream: STAGES boxes of (BOX_ROWS, 64) per CTA."""

    def __init__(self, stages: int, box_rows: int):
        self.stages = stages
        self.box_rows = box_rows

    @cute.jit
    def __call__(self, mBuf: cute.Tensor, mOut: cute.Tensor, stream):
        swizzle_128B = cute.make_swizzle(3, 4, 3)
        slayout = cute.make_composed_layout(
            swizzle_128B,
            0,
            cute.make_layout(
                (self.box_rows, COLS, self.stages),
                stride=(COLS, 1, self.box_rows * COLS),
            ),
        )
        tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mBuf,
            slayout,
            (self.box_rows, COLS),
        )
        smem_bytes = self.box_rows * COLS * self.stages * 2 + self.stages * 8 + 1024
        self.kernel(tma, mOut).launch(
            grid=(
                cute.size(mBuf, mode=[0]) // (self.stages * self.box_rows),
                1,
                1,
            ),
            block=(128, 1, 1),
            smem=smem_bytes,
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, tma: cpasync.TmaInfo, mOut: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        ST: cutlass.Constexpr = self.stages
        BR: cutlass.Constexpr = self.box_rows

        smem = cutlass.utils.SmemAllocator()
        sS = smem.allocate_tensor(
            BFloat16,
            tma.smem_layout.outer,
            byte_alignment=1024,
            swizzle=tma.smem_layout.inner,
        )
        mbar = smem.allocate_array(Int64, ST)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(ST):
                    cute.arch.mbarrier_init(mbar + i, 1)
        elif warp_idx == 1:
            cpasync.prefetch_descriptor(tma.atom)
        cute.arch.barrier()

        if warp_idx == 0:
            gTiles = cute.local_tile(tma.tma_tensor, (BR, COLS), (None, None))
            for kt in cutlass.range_constexpr(ST):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        mbar + kt, Int32(BR * COLS * 2)
                    )
                simple_tma_copy(
                    tma.atom,
                    gTiles[None, None, bid * ST + kt, 0],
                    sS[None, None, kt],
                    mbar + kt,
                )
            for kt in cutlass.range_constexpr(ST):
                cute.arch.mbarrier_wait(mbar + kt, 0)
        cute.arch.barrier()
        if tid == 0:
            mOut[bid] = sS[0, 0, 0]


class StreamLDG:
    """LDG.128 baseline: THREADS threads, U independent 16B loads in flight."""

    def __init__(self, kb_per_cta: int, unroll: int, threads: int = 128):
        self.kb = kb_per_cta
        self.unroll = unroll
        self.threads = threads
        self.vecs_per_cta = kb_per_cta * 1024 // 16

    @cute.jit
    def __call__(self, mVec: cute.Tensor, mOut: cute.Tensor, stream):
        # Force a static-stride (8,1) view so cute.copy can vectorize to
        # 16B (the fake tensor's symbolic strides defeat vectorization).
        mV = cute.make_tensor(
            mVec.iterator,
            cute.make_layout((cute.size(mVec, mode=[0]), 8), stride=(8, 1)),
        )
        self.kernel(mV, mOut).launch(
            grid=(cute.size(mVec, mode=[0]) // self.vecs_per_cta, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(self, mVec: cute.Tensor, mOut: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        U: cutlass.Constexpr = self.unroll
        TH: cutlass.Constexpr = self.threads
        VPC: cutlass.Constexpr = self.vecs_per_cta
        ITERS: cutlass.Constexpr = VPC // (U * TH)

        atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
        )
        rG = cute.make_rmem_tensor(cute.make_layout((U, 8), stride=(8, 1)), BFloat16)
        acc = Float32(0.0)
        base = bid * VPC + tid
        for i in cutlass.range_constexpr(ITERS):
            for u in cutlass.range_constexpr(U):
                cute.copy(atom, mVec[base + (i * U + u) * TH, None], rG[u, None])
            for u in cutlass.range_constexpr(U):
                for j in cutlass.range_constexpr(8):
                    acc += rG[u, j].to(Float32)
        mOut[bid * TH + tid] = acc.to(BFloat16)


def _compile_tma(stages, box_rows):
    from quack.compile_utils import make_fake_tensor

    r = cute.sym_int()
    o = cute.sym_int()
    buf = make_fake_tensor(BFloat16, (r, COLS), divisibility=8)
    out = make_fake_tensor(BFloat16, (o,), divisibility=8)
    return cute.compile(
        StreamTMA(stages, box_rows),
        buf,
        out,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _compile_ldg(kb, unroll):
    from quack.compile_utils import make_fake_tensor

    v = cute.sym_int()
    o = cute.sym_int()
    vec = make_fake_tensor(BFloat16, (v, 8), divisibility=8)
    out = make_fake_tensor(BFloat16, (o,), divisibility=8)
    return cute.compile(
        StreamLDG(kb, unroll),
        vec,
        out,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _dur_us(fn, flush_buf):
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
    warm_buf = torch.empty(1, device=DEVICE, dtype=torch.int8)  # no-op flush

    buf = torch.randn(BUF_BYTES // 2, device=DEVICE, dtype=DTYPE)
    out = torch.empty(296 * 128, device=DEVICE, dtype=DTYPE)
    rows = []

    def run(tag, per_cta_kb, g, fn, warm=False):
        total_kb = per_cta_kb * g
        if total_kb * 1024 > BUF_BYTES:
            return
        dur = _dur_us(fn, warm_buf if warm else flush_buf)
        gbs = total_kb * 1024 / dur / 1e3
        row = dict(
            cfg=tag,
            per_cta_kb=per_cta_kb,
            g=g,
            total_mb=total_kb / 1024,
            warm=warm,
            dur_us=round(dur, 2),
            gbs=round(gbs, 1),
        )
        rows.append(row)
        print(
            f"{tag:28s} {per_cta_kb:4d}KB/CTA G={g:3d} "
            f"tot={total_kb / 1024:6.2f}MB {'warm' if warm else 'cold'} "
            f"{dur:7.2f} us  {gbs:7.1f} GB/s",
            flush=True,
        )

    # (stages, box_rows): box_bytes = box_rows * 128
    tma_cfgs = [(5, 128), (10, 64), (5, 64), (10, 128), (2, 128), (20, 32)]
    for stages, box_rows in tma_cfgs:
        per_cta_kb = stages * box_rows * 128 // 1024
        kern = _compile_tma(stages, box_rows)
        for g in G_LIST:
            run(
                f"tma s{stages}xb{box_rows * 128 // 1024}k",
                per_cta_kb,
                g,
                lambda g=g, k=kern, ppc=stages * box_rows: k(
                    buf[: g * ppc * COLS].view(-1, COLS), out[:g]
                ),
            )
        run(
            f"tma s{stages}b{box_rows * 128 // 1024}k WARM",
            per_cta_kb,
            148,
            lambda k=kern, ppc=stages * box_rows: k(
                buf[: 148 * ppc * COLS].view(-1, COLS), out[:148]
            ),
            warm=True,
        )

    for kb, u in ((80, 8), (80, 16), (40, 8), (160, 8)):
        kern = _compile_ldg(kb, u)
        vpc = kb * 1024 // 16
        for g in G_LIST:
            run(
                f"ldg u{u}",
                kb,
                g,
                lambda g=g, k=kern, vpc=vpc: k(
                    buf[: g * vpc * 8].view(-1, 8), out[: g * 128]
                ),
            )
        run(
            f"ldg u{u} WARM",
            kb,
            148,
            lambda k=kern, vpc=vpc: k(
                buf[: 148 * vpc * 8].view(-1, 8), out[: 148 * 128]
            ),
            warm=True,
        )

    meta = dict(
        gpu=torch.cuda.get_device_name(0),
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
    )
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results_stream_floor.json"
    )
    with open(out_path, "w") as f:
        json.dump(dict(meta=meta, rows=rows), f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
