# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v7 smoke: PDL variant + strided-a (row stride 336) correctness."""

import torch
import torch.nn.functional as F

from workspace.qwen4_exp_mhc_llgemm.fused_up._up_gemm_v7 import UpGemmV7

K, N = 320, 10240
torch.cuda.set_device(0)
k7p = UpGemmV7(use_pdl=True)
k7 = UpGemmV7(use_pdl=False)
for m in [1, 8, 33, 64, 128]:
    torch.manual_seed(m)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    a_s = torch.randn(m, 336, device="cuda", dtype=torch.bfloat16)[:, :K]
    ref = F.linear(a, w)
    ref_s = F.linear(a_s, w)
    for tag, out, r in [
        ("pdl", k7p(a, w), ref),
        ("strided", k7(a_s, w), ref_s),
        ("pdl+strided", k7p(a_s, w), ref_s),
    ]:
        md = (out.float() - r.float()).abs().max().item()
        ok = torch.allclose(out.float(), r.float(), atol=0.6, rtol=0.02)
        print(f"M={m:4d} {tag:12s} md={md:.4f} {'OK' if ok else 'FAIL'}", flush=True)
        assert ok
print("all ok")
