# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick config probe for the CuteDSL gate_mix."""

from __future__ import annotations

import torch

import workspace.qwen4_exp_mhc_llgemm.bench_v5_hc_chain as bc
from vllm.models.qwen4_exp.nvidia.ops.hc import hc_gate_mix
from workspace.qwen4_exp_mhc_llgemm.gate_mix_cute import GateMixCute

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16
HC, HD = 4, 2560
DIM = HC * HD


def main() -> None:
    torch.cuda.set_device(0)
    from flashinfer.testing.utils import get_l2_cache_size

    flush_buf = torch.empty(
        2 * get_l2_cache_size(DEVICE), device=DEVICE, dtype=torch.int8
    )
    kernels = {}
    for m in [1, 128, 512]:
        torch.manual_seed(0)
        xn = torch.randn(m, DIM, device=DEVICE, dtype=DTYPE)
        gate = torch.randn(m, DIM, device=DEVICE, dtype=DTYPE)
        ref = hc_gate_mix(xn, gate, HC)
        cfgs = {
            "prod_triton": (lambda: hc_gate_mix(xn, gate, HC)),
        }
        for vec, thr, pdl, sig in [
            (8, 256, True, "exp"),
            (8, 256, False, "tanh"),
            (4, 128, False, "exp"),
            (4, 128, False, "tanh"),
            (4, 128, False, "exp2"),
            (4, 128, False, "identity"),
            (8, 128, False, "tanh"),
        ]:
            key = (vec, thr, pdl, sig)
            if key not in kernels:
                kernels[key] = GateMixCute(
                    vec=vec, threads=thr, use_pdl=pdl, sigmoid_mode=sig
                )
            k = kernels[key]
            out = k(xn, gate)
            md = (out.float() - ref.float()).abs().max().item()
            if sig != "identity":
                assert md < 2e-2, (key, m, md)
            cfgs[f"cute_v{vec}_t{thr}_{sig}{'_pdl' if pdl else ''}"] = lambda k=k: k(
                xn, gate
            )
        for tag, fn in cfgs.items():
            iters_k = bc.measure_graph(fn, flush_buf)
            summ, _ = bc.summarize(iters_k)
            print(f"M={m:4d} {tag:22s} {summ['_total_us']:6.2f}us", flush=True)


if __name__ == "__main__":
    main()
