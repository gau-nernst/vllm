# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton hc_gate_mix improvement sweep (baseline hygiene, M range-wide).

Variants vs the production kernel (vllm/models/qwen4_exp/nvidia/ops/hc.py):
  prod          production _hc_gate_mix_kernel (masked, BLOCK=512, serial streams)
  serial        same structure, mask dropped (HC_DIM % BLOCK == 0 statically)
  materialized  all HC streams loaded as one [HC, BLOCK] tile, tl.sum over axis 0

Sweep: BLOCK x num_warps, M list. Cold-L2 CUDA-graph CUPTI medians, PDL on
(gdc_wait / gdc_launch_dependents exactly like production).

    PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
        workspace/qwen4_exp_mhc_llgemm/bench_gate_mix.py
"""

from __future__ import annotations

import sys

import torch

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
from vllm.triton_utils import tl, triton

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD = 4, 2560
DIM = HC * HD
M_LIST = [1, 8, 32, 96, 128, 256, 384, 512]


@triton.jit
def _gm_serial(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for stream in tl.static_range(HC):
        o = stream * HC_DIM + offs
        g = tl.load(g_ptr + row * stride_g + o)
        x = tl.load(x_ptr + row * stride_x + o)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs, acc.to(y_ptr.dtype.element_ty))


@triton.jit
def _gm_lookahead(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    # Serial accumulate (prod order) but all 2*HC bf16 loads issued before any
    # dependent math: no [HC, BLOCK] fp32 materialization, no load->acc chains.
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    g0 = tl.load(g_ptr + row * stride_g + offs)
    x0 = tl.load(x_ptr + row * stride_x + offs)
    g1 = tl.load(g_ptr + row * stride_g + HC_DIM + offs)
    x1 = tl.load(x_ptr + row * stride_x + HC_DIM + offs)
    g2 = tl.load(g_ptr + row * stride_g + 2 * HC_DIM + offs)
    x2 = tl.load(x_ptr + row * stride_x + 2 * HC_DIM + offs)
    g3 = tl.load(g_ptr + row * stride_g + 3 * HC_DIM + offs)
    x3 = tl.load(x_ptr + row * stride_x + 3 * HC_DIM + offs)
    acc = tl.sigmoid(g0.to(tl.float32)) * x0.to(tl.float32)
    acc += tl.sigmoid(g1.to(tl.float32)) * x1.to(tl.float32)
    acc += tl.sigmoid(g2.to(tl.float32)) * x2.to(tl.float32)
    acc += tl.sigmoid(g3.to(tl.float32)) * x3.to(tl.float32)
    acc /= HC
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs, acc.to(y_ptr.dtype.element_ty))


@triton.jit
def _gm_materialized(
    x_ptr,
    g_ptr,
    y_ptr,
    stride_x,
    stride_g,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK: tl.constexpr,
    launch_pdl: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tile * BLOCK + tl.arange(0, BLOCK)
    offs2 = tl.arange(0, HC)[:, None] * HC_DIM + offs[None, :]
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    g = tl.load(g_ptr + row * stride_g + offs2)
    x = tl.load(x_ptr + row * stride_x + offs2)
    acc = tl.sum(tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32), axis=0) / HC
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(y_ptr + row * stride_y + offs, acc.to(y_ptr.dtype.element_ty))


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    m_list = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else M_LIST
    l2 = get_l2_cache_size(DEVICE)
    flush_buf = torch.empty(2 * l2, device=DEVICE, dtype=torch.int8)

    variants = {"prod": None}
    for blk, nw in [(256, 2), (256, 4), (512, 2), (512, 4), (512, 8)]:
        variants[f"serial_b{blk}_w{nw}"] = (_gm_serial, blk, nw)
    for blk, nw in [(256, 4), (512, 2), (512, 4), (512, 8)]:
        variants[f"lookah_b{blk}_w{nw}"] = (_gm_lookahead, blk, nw)

    for m in m_list:
        torch.manual_seed(0)
        xn = torch.randn(m, DIM, device=DEVICE, dtype=DTYPE)
        gate = torch.randn(m, DIM, device=DEVICE, dtype=DTYPE)
        ref = hc_gate_mix(xn, gate, HC)
        for tag, cfg in variants.items():
            if cfg is None:
                fn = lambda: hc_gate_mix(xn, gate, HC)
            else:
                kern, blk, nw = cfg

                def fn(kern=kern, blk=blk, nw=nw):
                    out = xn.new_empty(m, HD)
                    kern[(m, HD // blk)](
                        xn,
                        gate,
                        out,
                        xn.stride(0),
                        gate.stride(0),
                        out.stride(0),
                        HD,
                        HC,
                        blk,
                        True,
                        num_warps=nw,
                    )
                    return out

            out = fn()
            md = (out.float() - ref.float()).abs().max().item()
            # serial/lookahead preserve prod's accumulate order; allow 1-ulp
            # bf16 flips from compiler reassociation (seen: 0.002 at M>=128).
            assert md < 3e-3, (tag, m, md)
            iters_k = bc.measure_graph(fn, flush_buf)
            summ, _ = bc.summarize(iters_k)
            print(f"M={m:3d} {tag:20s} {summ['_total_us']:.2f}", flush=True)


if __name__ == "__main__":
    main()
