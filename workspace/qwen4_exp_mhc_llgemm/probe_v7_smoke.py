# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""v7 smoke: correctness vs torch for a few M, incl. grid.y spill."""

import torch
import torch.nn.functional as F

from workspace.qwen4_exp_mhc_llgemm.fused_up._up_gemm_v7 import UpGemmV7

K, N = 320, 10240
torch.cuda.set_device(0)
k7 = UpGemmV7(use_pdl=False)
for m in [1, 8, 33, 64, 128]:
    torch.manual_seed(m)
    a = torch.randn(m, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(a, w)
    out = k7(a, w)
    md = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=0.6, rtol=0.02)
    print(f"M={m:4d} md={md:.4f} {'OK' if ok else 'FAIL'}", flush=True)
    assert ok
print("all ok")
