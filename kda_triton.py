# ruff: noqa: B023, E741
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
import math
import random
import statistics
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.model_executor.layers.fla.ops.kda import chunk_kda_fwd
from vllm.triton_utils import tl, triton

SCRIPT_DIR = Path(__file__).resolve().parent
for local_repo in ("cuLA", "FlashKDA"):
    local_path = str(SCRIPT_DIR / local_repo)
    if local_path not in sys.path:
        sys.path.insert(0, local_path)
fla_path = str(SCRIPT_DIR / "cuLA" / "third_party" / "flash-linear-attention")
if fla_path not in sys.path:
    sys.path.insert(0, fla_path)

try:
    from cula.kda.chunk_fwd import chunk_kda_fwd as cula_chunk_kda_fwd
    from fla.ops.utils.index import prepare_chunk_indices
except Exception:
    cula_chunk_kda_fwd = None

try:
    import flash_kda
except Exception:
    flash_kda = None


HEAD_K_DIM = 128
HEAD_V_DIM = 128
KDA_LOWER_BOUND = -5.0
MODEL_CASES = (
    ("TP1", 64),
    ("TP4", 16),
)
SEQ_CASES = (
    ((8192,), "1x8192"),
    ((16384,), "1x16384"),
    ((1024,) * 8, "8x1024"),
    ((2048,) * 8, "8x2048"),
    ((1024,) * 16, "16x1024"),
)


def make_mixed_seqlens(
    num_requests: int,
    total_tokens: int,
    seed: int,
) -> tuple[int, ...]:
    rng = random.Random(seed)
    weights = [rng.expovariate(1.0) ** 2 for _ in range(num_requests)]
    scale = (total_tokens - num_requests) / sum(weights)
    seqlens = [1 + int(weight * scale) for weight in weights]
    seqlens[-1] += total_tokens - sum(seqlens)
    return tuple(seqlens)


MIXED_SEQ_CASES = (
    (make_mixed_seqlens(8, 8192, seed=8), "8@8192"),
    (make_mixed_seqlens(16, 8192, seed=16), "16@8192"),
    (make_mixed_seqlens(16, 16384, seed=16384), "16@16384"),
    (make_mixed_seqlens(32, 16384, seed=32), "32@16384"),
)
ALL_SEQ_CASES = SEQ_CASES + MIXED_SEQ_CASES


@triton.jit
def kda_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    beta_ptr,
    g_ptr,
    h0_ptr,
    out_ptr,
    ht_ptr,
    cu_seqlens_ptr,
    scale: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BT: tl.constexpr,
):
    seq_id = tl.program_id(0)
    head = tl.program_id(1)

    bos = tl.load(cu_seqlens_ptr + seq_id)
    eos = tl.load(cu_seqlens_ptr + seq_id + 1)

    offs_t = tl.arange(0, BT)
    offs_k = tl.arange(0, DK)
    offs_v = tl.arange(0, DV)

    I = (offs_t[:, None] == offs_t).to(tl.float32)
    causal = offs_t[:, None] >= offs_t
    strict_lower = offs_t[:, None] > offs_t

    # H maps key/state dimension to value dimension: [DV, DK].
    h_f32 = tl.load(
        h0_ptr + seq_id * H * DV * DK + head * DV * DK + offs_v[:, None] * DK + offs_k
    ).to(tl.float32)

    for chunk_start in range(bos, eos, BT):
        token = chunk_start + offs_t
        token_mask = token < eos
        token_pair_mask = token_mask[:, None] & token_mask

        k_offs = token[:, None] * H * DK + head * DK + offs_k
        q = tl.load(q_ptr + k_offs, mask=token_mask[:, None], other=0.0)
        k = tl.load(k_ptr + k_offs, mask=token_mask[:, None], other=0.0)
        g = tl.load(g_ptr + k_offs, mask=token_mask[:, None], other=0.0)
        g_cu = tl.cumsum(g, 0)
        g_last = tl.sum(g, axis=0)

        v_offs = token[:, None] * H * DV + head * DV + offs_v
        v = tl.load(v_ptr + v_offs, mask=token_mask[:, None], other=0.0)
        beta = tl.load(beta_ptr + token * H + head, mask=token_mask, other=0.0)

        # kkt
        k_left = (k * tl.exp(g_cu)).to(tl.bfloat16)
        k_right = (k * tl.exp(-g_cu)).to(tl.bfloat16)
        kkt = tl.dot(k_left, k_right.T)
        A = tl.where(strict_lower & token_pair_mask, beta[:, None] * kkt, 0.0)

        # inv(I + A) using Newton-Schulz iterations.
        M = I + A
        Ai = I - A
        NS_ITERS: tl.constexpr = int(math.log2(BT)) - 1
        for _ in tl.static_range(NS_ITERS):
            Ai = 2 * Ai - tl.dot(tl.dot(Ai, M), Ai)

        Aib = (Ai * beta).to(tl.bfloat16)
        u = tl.dot(Aib, v).to(tl.bfloat16)
        w = tl.dot(Aib, k_left).to(tl.bfloat16)

        h_bf16 = h_f32.to(tl.bfloat16)
        v_new = (u - tl.dot(w, h_bf16.T)).to(tl.bfloat16)

        h_f32 = tl.dot(v_new.T, k_right, acc=h_f32) * tl.exp(g_last)

        q_left = (q * (tl.exp(g_cu) * scale)).to(tl.bfloat16)
        qk = tl.dot(q_left, k_right.T)
        p = tl.where(causal & token_pair_mask, qk, 0.0)
        pv = tl.dot(p.to(tl.bfloat16), v_new)

        qh = tl.dot(q_left, h_bf16.T)
        o = qh + pv
        tl.store(out_ptr + v_offs, o, mask=token_mask[:, None])

    tl.store(
        ht_ptr + seq_id * H * DV * DK + head * DV * DK + offs_v[:, None] * DK + offs_k,
        h_f32,
    )


def kda_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    beta: Tensor,
    g: Tensor,
    h0: Tensor,
    cu_seqlens: Tensor,
    *,
    chunk_size: int = 16,
) -> tuple[Tensor, Tensor]:
    _, H, DK = q.shape
    _, _, DV = v.shape
    batch = cu_seqlens.numel() - 1
    out = torch.empty_like(v)
    ht = torch.empty_like(h0)

    kda_prefill_kernel[(batch, H)](
        q,
        k,
        v,
        beta,
        g,
        h0,
        out,
        ht,
        cu_seqlens,
        float(DK**-0.5),
        H,
        DK,
        DV,
        chunk_size,
        num_warps=4,
    )
    return out, ht


def make_inputs(
    lens: tuple[int, ...],
    *,
    H: int,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    torch.manual_seed(seed)
    total_tokens = sum(lens)
    batch = len(lens)
    dtype = torch.bfloat16

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, dtype=dtype)

    q = F.normalize(randn(1, total_tokens, H, HEAD_K_DIM).float(), p=2, dim=-1).to(
        dtype
    )
    k = F.normalize(randn(1, total_tokens, H, HEAD_K_DIM).float(), p=2, dim=-1).to(
        dtype
    )
    v = randn(1, total_tokens, H, HEAD_V_DIM)
    raw_g = randn(1, total_tokens, H, HEAD_K_DIM)

    A = torch.empty(H, dtype=torch.float32).uniform_(0.0, 16.0)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(H * HEAD_K_DIM, dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    beta_logits = torch.randn(1, total_tokens, H, dtype=dtype)
    h0 = randn(batch, H, HEAD_V_DIM, HEAD_K_DIM).float()
    cu_seqlens = F.pad(torch.tensor(lens, dtype=torch.int32).cumsum(0), (1, 0)).to(
        torch.int32
    )
    return q, k, v, h0, cu_seqlens, raw_g, A_log, dt_bias, beta_logits


def compute_gate(raw_g: Tensor, A_log: Tensor, dt_bias: Tensor) -> Tensor:
    H = raw_g.shape[2]
    return KDA_LOWER_BOUND * torch.sigmoid(
        torch.exp(A_log.view(1, 1, H, 1))
        * (raw_g.float() + dt_bias.view(H, HEAD_K_DIM))
    )


def mae(actual: Tensor, expected: Tensor) -> float:
    return float((actual.float() - expected.float()).abs().mean())


def benchmark_kda_prefill() -> None:
    from flashinfer.testing import bench_gpu_time_with_cupti

    torch.set_default_device("cuda")
    torch.manual_seed(int(time.time()))

    print("KDA prefill benchmark")
    print("- baseline: FLA (vLLM chunk_kda_fwd)")
    print("- timing: CUPTI median; speedup is relative to FLA")
    print("- MAE: output / final-state vs FLA")
    print("- gate/beta activation is precomputed for FLA, Triton dev, and cuLA")
    print("- FlashKDA computes gate, beta, and q/k L2 norm inside its prepare kernel")
    print("- dtype: bfloat16")
    print("- Kimi Linear KDA dims: H=32 total, DK=DV=128")
    print("- mixed sequence lengths:")
    for seqlens, seq_name in MIXED_SEQ_CASES:
        print(f"  - {seq_name}: {seqlens}")

    timing_rows: list[dict[str, object]] = []
    mae_rows: list[dict[str, object]] = []
    seed = 1000
    for (model_name, H), (lens, seq_name) in itertools.product(
        MODEL_CASES, ALL_SEQ_CASES
    ):
        q, k, v, h0, cu_seqlens, raw_g, A_log, dt_bias, beta_logits = make_inputs(
            lens, H=H, seed=seed
        )
        seed += 1

        gate = compute_gate(raw_g, A_log, dt_bias)
        beta = beta_logits.sigmoid()

        # Each backend gets private buffers outside the timed region. Some paths
        # reuse v as output storage and update final-state buffers.
        v_fla = v.clone()
        v_triton = v.clone()
        v_cula = v.clone()
        v_flash = v.clone()
        h0_fla = h0.clone()
        h0_triton = h0.clone()
        h0_cula = h0.transpose(-1, -2).clone(memory_format=torch.contiguous_format)
        h0_flash = h0.clone()

        def run_fla() -> tuple[Tensor, Tensor]:
            return chunk_kda_fwd(
                q,
                k,
                v_fla,
                gate,
                beta,
                float(HEAD_K_DIM**-0.5),
                initial_state=h0_fla,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
            )

        def run_triton() -> tuple[Tensor, Tensor]:
            out, ht = kda_prefill(
                q.squeeze(0),
                k.squeeze(0),
                v_triton.squeeze(0),
                beta.squeeze(0),
                gate.squeeze(0),
                h0_triton,
                cu_seqlens,
            )
            return out.unsqueeze(0), ht

        chunk_indices = prepare_chunk_indices(cu_seqlens, 64).to(torch.int32)

        def run_cula() -> tuple[Tensor, Tensor]:
            out = cula_chunk_kda_fwd(
                q=q,
                k=k,
                v=v_cula,
                g=gate,
                beta=beta,
                scale=float(HEAD_K_DIM**-0.5),
                initial_state=h0_cula,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                safe_gate=True,
                lower_bound=KDA_LOWER_BOUND,
                use_gate_in_kernel=False,
            )
            return out[0], out[1]

        cu_seqlens_i64 = cu_seqlens.to(torch.int64)
        dt_bias_2d = dt_bias.view(A_log.numel(), HEAD_K_DIM)

        def run_flash() -> tuple[Tensor, Tensor]:
            out = torch.empty_like(v_flash)
            final_state = torch.empty_like(h0_flash)
            flash_kda.fwd(
                q,
                k,
                v_flash,
                raw_g,
                beta_logits,
                float(HEAD_K_DIM**-0.5),
                out,
                A_log,
                dt_bias_2d,
                KDA_LOWER_BOUND,
                initial_state=h0_flash,
                final_state=final_state,
                cu_seqlens=cu_seqlens_i64,
            )
            return out, final_state

        ref_o, ref_ht = run_fla()
        ref_o = ref_o.clone()
        ref_ht = ref_ht.clone()
        triton_o, triton_ht = run_triton()
        triton_o = triton_o.clone()
        triton_ht = triton_ht.clone()
        if cula_chunk_kda_fwd is not None:
            cula_o, cula_ht = run_cula()
            cula_o = cula_o.clone()
            cula_ht = cula_ht.transpose(-1, -2).clone()
        if flash_kda is not None:
            flash_o, flash_ht = run_flash()
            flash_o = flash_o.clone()
            flash_ht = flash_ht.clone()
        torch.accelerator.synchronize()

        fla_ms = statistics.median(bench_gpu_time_with_cupti(run_fla))
        triton_ms = statistics.median(bench_gpu_time_with_cupti(run_triton))
        if cula_chunk_kda_fwd is not None:
            cula_ms = statistics.median(bench_gpu_time_with_cupti(run_cula))
        if flash_kda is not None:
            flash_ms = statistics.median(bench_gpu_time_with_cupti(run_flash))

        timing_row: dict[str, object] = {
            "Model": model_name,
            "Seq": seq_name,
            "H": H,
            "FLA": f"{fla_ms:.3f} ms",
            "Triton dev": f"{triton_ms:.3f} ms ({fla_ms / triton_ms:.2f}x)",
        }
        mae_row: dict[str, object] = {
            "Model": model_name,
            "Seq": seq_name,
            "H": H,
            "Triton dev": f"{mae(triton_o, ref_o):.4g} / {mae(triton_ht, ref_ht):.4g}",
        }
        if cula_chunk_kda_fwd is not None:
            timing_row["cuLA"] = f"{cula_ms:.3f} ms ({fla_ms / cula_ms:.2f}x)"
            mae_row["cuLA"] = f"{mae(cula_o, ref_o):.4g} / {mae(cula_ht, ref_ht):.4g}"
        if flash_kda is not None:
            timing_row["FlashKDA"] = f"{flash_ms:.3f} ms ({fla_ms / flash_ms:.2f}x)"
            mae_row["FlashKDA"] = (
                f"{mae(flash_o, ref_o):.4g} / {mae(flash_ht, ref_ht):.4g}"
            )
        timing_rows.append(timing_row)
        mae_rows.append(mae_row)
        torch.accelerator.empty_cache()

    print("\nTiming")
    print(pd.DataFrame(timing_rows).to_markdown(index=False))
    print("\nMAE (O / H vs FLA)")
    print(pd.DataFrame(mae_rows).to_markdown(index=False))
    print()


if __name__ == "__main__":
    benchmark_kda_prefill()
