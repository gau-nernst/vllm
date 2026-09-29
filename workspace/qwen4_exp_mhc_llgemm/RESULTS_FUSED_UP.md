# Fused mHC UP kernel (GB300): gate GEMM + sigmoid-mix epilogue

Metadata: GPU = NVIDIA GB300 (SM103), bf16, commit
`3137ff07739f0cd19e06119506c6432a4a7e77ad`, torch 2.13.0+cu130, cupti-python
13.4.0. Timing: median CUPTI GPU time, CUDA graph, cold L2 (rotating input
buffers), 25 warmup iters; all kernels compiled before timing.

Kernel: `fused_up/` (`_up_gate_mix.py` = v1 "warp-per-channel",
`_up_gate_mix_v2.py` = v2 "token x k-slice", `_up_gate_mix_v3.py` = v3
"activation-stationary M-blocked", `up_gate_mix.py` = wrapper, `impl=`
selects v1/v2/v3).
Semantics fused into the UP GEMM (K=320, N=10240, HC=4 → out [M, 2560]):

    out[m, h] = (1/HC) * sum_s sigmoid(bf16(gate_s[m])) * xn[m, s*2560 + h]
        gate_s[m] = dot(a[m, :], w[s*2560 + h, :])   (fp32 accumulation)

The bf16 rounding boundary of the production GEMM output is preserved
(fp32 dot → bf16 → fp32 → sigmoid); the [M, N] gate tensor is never
materialized. Production chain = `hc_gate_mix(xn, F.linear(a, w), 4)`
(Triton epilogue kernel in `vllm/models/qwen4_exp/nvidia/ops/hc.py`).

v1 design (the shipped default): one CTA per CH=4 channels (640 CTAs), one
warp per channel; warp preloads its channel's 4 weight rows into registers
(K=320 = 32 lanes x vec8 main + 8-lane ragged tail via constexpr-offset tail
views), then loops over tokens: 2x16B `a`-row load, 4 warp-local dots + full
warp shuffle reductions, gates staged via SMEM, `xn` prefetched before the
token loop, thread-per-(token, channel) epilogue. M dynamic (≤ 256).

## Correctness (per M, before timing)

- vs production chain: **bit-identical (maxdiff 0.0) at M ∈ {1,2,4}**;
  0.00024 @ M=8, 0.00049 @ M=16/32, 0.00195 @ M=48..128 (bf16 ulp-level,
  within the 2e-2 fused-chain tolerance).
- vs independent fp32 reference: ≤ 0.0039 at M ≤ 64, 0.0071 at M=128/256.
- M=256 correctness-only probe passes (no hard M limit in v1).

## Timing (µs), fused v1 vs unfused chains

| M   | GEMM only (cuBLAS) | best unfused chain | fused (v1) | speedup vs chain | fused GB/s |
|-----|--------------------|--------------------|------------|------------------|------------|
| 1   | 4.51 | 6.18 (cublas)   | **5.82** | **1.06×** | 1130 |
| 2   | 4.51 | 6.24 (cublas)   | 6.62  | 0.94× | 997 |
| 4   | 4.51 | 6.30 (cublasLt) | 8.48  | 0.74× | 785 |
| 8   | 4.48 | 6.34 (cublasLt) | 12.06 | 0.53× | 561 |
| 16  | 4.86 | 6.75 (cublas)   | 21.38 | 0.32× | 326 |
| 32  | 4.93 | 7.04 (cublasLt) | 39.04 | 0.18× | 189 |
| 48  | 4.67 | 8.93 (cublas)   | 59.17 | 0.15× | 132 |
| 64  | 4.77 | 8.90 (cublasLt) | 83.04 | 0.11× | 99 |
| 128 | 5.18 | 9.38 (cublasLt) | 167.66| 0.06× | 59 |

(cuBLAS vs cuBLASLt are within noise of each other at every M.)

## Conclusion

**UP fusion pays only at M=1** (5.82 µs vs 6.18 µs chain, 1.06×, bit-identical
— saves the `hc_gate_mix` launch + [M,N] gate round-trip). At M≥2 the fused
kernel loses, increasingly badly: its per-token cost is ~1.1 µs while cuBLAS
adds ~0.01 µs/token.

Why the fused kernel can't scale (measured, not guessed):

- *Scalar-FFMA ceiling.* The CuTe-DSL dot uses fp32 FFMA (no tensor-core path
  for this shape in the DSL). The pure-FFMA floor at M=128 is ~12.6 µs
  (419M MACs / (148 SMs × 128 FMA/cyc)) — already above the entire unfused
  chain (9.38 µs), so large-M fusion can never win on this hardware.
- *Per-token DRAM-latency serialization.* With cold L2, each token's `a`-row
  load is a ~1500-cycle DRAM access on the critical path (load → dot → shuffle
  reduce → sigmoid → store). Ablation on v2 at M=128: loads+dot only = 23.9 µs,
  full = 261 µs → the gate→epilogue dependency chain dominates.
- *v2 (token×k-slice lanes, no SMEM/barrier, hoisted `xn` loads, 3-step
  butterfly reduces)* fixed the structural serialization but not the latency
  exposure, and its higher per-lane work/registers (234 vs 136) hurt the M=1
  latency case: v2 = 8.9 µs @M=1 vs v1 = 6.2 µs (pre-unroll-tuning). m-loop
  unroll sweep on v1: unroll=1 best at M=1 (5.82–5.89 µs stable), unroll=4
  best at M≥8 (M=32: 36.4→27.9 µs) but still far behind the chain.

SASS lessons (apply to any CuTe-DSL elementwise/GEMM kernel here):

- `cute.autovec_copy` into a **slice of a larger rmem tensor**
  (`wr[s, 0, None]`) scalarizes the global loads to 8×LDG.E.U16; copying into a
  standalone `make_rmem_tensor_like(gmem_slice)` fragment keeps LDG.E.128.
  Fixing this in v1: 7.30 → 6.21 µs @M=1.
- Ragged K tails: `lane + WARP` dynamic indexing scalarizes; constexpr
  `cute.domain_offset((0, WARP*VEC), ...)` tail views keep vectorization.

Recommendation: keep the production unfused UP chain (cuBLAS + `hc_gate_mix`);
optionally dispatch to `fused_up.up_gate_mix_gemm` only for M=1 decode
(~0.35 µs/token saved, ~5% of the UP stage). The DOWN-side fusion
(merged upstream, PR #58957) remains the worthwhile one.

Reproduce:

```bash
cd /home/thien/vllm/main && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python workspace/qwen4_exp_mhc_llgemm/bench_fused_up.py
```

Raw data: `results_fused_up.json`.

## v3: activation-stationary, weight-streaming, M-blocked (`_up_gate_mix_v3.py`)

Final config: 320-CTA grid (x4 M-split), 4 warps/CTA, **CW=2 channels per
warp**, MB=4 tokens per M-block (invariant CW*HC*MB=32 → one staging row per
lane). Weights for all CW*HC=8 rows preloaded to fp32 registers before the
PDL wait; activation K-fragments are loaded once per token per lane and
reused across all 8 weight fragments (amortizes the bf16→fp32 SHF converts
and LDGs over 2× more FFMAs). `a` is read straight from gmem (20KB, L1/L2
resident); the cross-lane reduce uses bank-conflict-free padded SMEM staging
- an in-register tree replicating `warp_reduction_sum`'s butterfly order;
epilogue keeps the production s-serial mix. The M-split grid (grid.y=4)
spreads M-blocks over 4 CTAs per channel group to raise warp supply at M≥8;
CTAs whose first block is past M skip the whole body (phantom CTAs would
otherwise double the cold w stream at small M — measured +1.7 µs at M=1).
Each dot is covered by 16 lanes (`lanes_per_dot=16`: two dot groups per
warp, interleaved k-slices), which halves the SMEM staging/reduce traffic
at a 2× activation-load duplication cost — measured faster at every M.
M dynamic (≤ 64, two compiled variants), CUDA-graph safe, PDL kept.

### Correctness (per M)

Bit-identical to the production chain at M ∈ {1,2} — this survives the
lanes_per_dot=16 reduction-order change (20-elem partials, 16-wide tree):
the fp32 ordering difference is far below the bf16 gate-rounding boundary.
maxdiff vs chain = 0.0078 at M ≥ 4 (cuBLAS
itself switches k-order at M ≥ 4, so bit-identity there is unreachable); vs
fp32 reference ≤ 0.0039 at all M (bf16 ulp-level, same class as v1).

### Timing (µs), fused v3 (final) vs unfused chains

| M   | best unfused chain | fused (v3 final) | speedup | v3 pre-pass | fused v1 |
|-----|--------------------|------------------|---------|-------------|----------|
| 1   | 6.14 (cublas)   | **4.64** | **1.32×** | 5.94  | 5.82 |
| 2   | 6.21 (cublasLt) | **4.64** | **1.34×** | 6.24  | 6.62 |
| 4   | 6.27 (cublas)   | **4.67** | **1.34×** | 6.18  | 8.48 |
| 8   | 6.34 (cublas)   | 6.56  | 0.97× | 6.72  | 12.06 |
| 16  | 6.75 (cublasLt) | 8.86  | 0.76× | 10.30 | 21.38 |
| 32  | 7.01 (cublas)   | 14.88 | 0.47× | 18.56 | 39.04 |
| 48  | 8.70 (cublas)   | 20.93 | 0.42× | 26.98 | 59.17 |
| 64  | 8.86 (cublas)   | 26.66 | 0.33× | 34.37 | 83.04 |

**Verdict: v3 (final) beats the unfused chain by 1.32–1.34× at M ≤ 4** —
solidly, and bit-identically at M ≤ 2 — and beats v1's M=1 (4.64 vs
5.82 µs), so unlike the pre-pass v3 it is *not* dominated: it is the best
fused option for M ∈ {1,2,4}. At M=8 it is 3% behind the chain (6.56 vs
6.34 µs, narrowed from 5% by lanes_per_dot=16), and M ≥ 16
still loses. The "beat the chain at every M ≤ 32" target was NOT met; see
the limiter analysis below.

### Follow-up tuning pass: what each change bought (or cost)

Baseline (pre-pass): 5.94/6.24/6.18/6.72/10.30/18.56 µs at M=1/2/4/8/16/32.

- **Register caps (maxrregcount ∈ {64,80,96}): no effect** (±0.1 µs, no
  harmful spills even at 64). The ncu "Block Limit Registers = 4 CTAs/SM"
  was a red herring: the grid only supplies 640 CTAs / 148 SMs = 4.3 CTAs/SM,
  so residency was grid-limited, not register-limited. v3 compiles with no
  regcap; the ll_bf16-style `-maxrregcount=64` does not help here.
- **channels_per_cta ∈ {2,8,16}, min_blocks_per_mp=4: no effect or worse**
  (CH=16: 23.2 µs at M=32). Total warp supply (2560 warps) is invariant
  under CH — only a decomposition change can raise it.
- **CW=2 register blocking over channels (the key win)**: each lane loads an
  activation K-fragment once per token and reuses it across CW×HC=8 weight
  fragments → SHF-cvt and LDG cost per FFMA halved, 32 independent FMA
  chains per lane. M=1: 5.94→4.43; M=4: 6.18→4.80; M=8: 6.72→6.56; M=32:
  18.56→17.95. CW=4/MB=2 (160 fp32 weight regs, 640 warps): better at M=1
  (4.32) but much worse at M≥8 (occupancy starvation) — rejected. CW=8 not
  viable (320 weight regs). Numerics unchanged (bit-identity at M≤2 kept).
- **M-split grid ×4 + phantom-CTA skip**: doubles/quadruples warp supply at
  M≥8 to cover gmem latency (ncu: long-scoreboard was the top stall, 3.06
  cyc/inst). Dots phase at M=32: 6.91→4.83 µs (vs ~3.3 µs FFMA-issue
  floor); M=16: 10.30→9.06; M=32: 17.46→15.12. Phantom CTAs must skip the
  weight stream (dynamic guard around the whole body — the DSL forbids
  dynamic early `return`); without the skip M=1 regresses 4.43→6.14 µs.
  DSL note: a big dynamic `if` wrapping gmem→rmem autovec_copies and direct
  SMEM stores compiles fine (the earlier "operand does not dominate" failure
  was specific to cp.async `cute.copy` into SMEM slices).
- **unroll=2 on the M-block loop**: 15.26→15.12 at M=32, 9.15→9.06 at M=16.
  Kept.
- **LDS.128 vectorized reduce (stride-36 padded rows): ~1 µs SLOWER at M=32
  — reverted.** SASS confirms 8×LDS.128 per row replace 32×LDS.32, yet
  dm0 regressed; the scalar stride-33 layout wins.
- **Halving-butterfly SHFL reduce: killed on analysis.** Warp-wide SIMD
  executes one SHFL per *value in flight* per stage regardless of the
  halving trick (32 values × 5 stages = 160 SHFL/block, same as the full
  butterfly that already lost to SMEM staging's 64 MIO insts).
- regcap {128,104,88} at the final config: no effect at M=8/32.
- **Packed f32x2 FMA (fma.rn.f32x2 / SASS FFMA2): achievable from the DSL
  but SLOWER — rejected.** The shipped kernel's dot loop emits scalar FFMA
  (SASS: 660 FFMA, 0 FFMA2; PTX: 972 `fma.rn.f32`, 0 `fma.rn.f32x2`). The
  construct that emits it: `cute.arch.fma_packed_f32x2((a, a), (w_s, w_s1),
  (acc_s, acc_s1))` (in `cute/arch/nvvm_wrappers.py`; quack uses it in
  `activation.py`) with the two accumulators of a stream pair advancing in
  one packed op and `a` broadcast into both halves — each accumulator keeps
  its exact serial v-order chain, so numerics are bit-identical. Result:
  PTX 480 `fma.rn.f32x2`, SASS 480 FFMA2 with proper `.F32x2.HI_LO`
  register-pair operands, same 168 regs, no packing MOVs. But measured: dm2
  (dots) @M=32 4.99→5.38 µs, full @M=1/4 4.58/4.64→4.80/4.90 µs, full @M=8
  6.62→6.85 µs; only M=32 full improved (15.12→14.21 µs, still 0.49× vs
  chain). Conclusion: on SM103 FFMA2 issues at ~the scalar FFMA rate (no FP32
  throughput doubling for this kernel) and the register-pair constraints
  slightly hurt scheduling. The dots phase did NOT beat the issue floor, so
  there was nothing to re-balance toward the epilogue. Scalar FFMA stays.
  (Loop `vectorize=True` was later tried too — it cannot express the
  pairing; and a structural TensorSSA pairing with zero packing MOVs
  confirmed FFMA2 does not dual-issue on SM103. See the final-final pass
  below.)

Final FFMA pass (m-split weight-stream sharing + sub-warp dots):

- **M-split grid reorder (m_split fastest-varying): rejected.** Same-channel
  CTAs adjacent in launch order: 4.90/4.96/6.62/8.86/14.37 µs at
  M=1/4/8/16/32 — helps M ≥ 16 slightly but regresses the M ≤ 4 win window
  (phantom-exit CTAs interleave between working msub=0 CTAs in launch
  order). Reverted; channel-group-fastest stays (see the launch-site NOTE).
- **.cg (L2-only) weight loads (`LoadCacheMode.GLOBAL`): rejected.** Weights
  are read once per CTA, so keeping them out of L1 should leave room for
  activations: measured neutral at M ≤ 4 (4.61/4.69) and worse at M=8/16
  (6.96/9.22). Reverted.
- **DSMEM clustering: skipped** — gated on the m-split experiments showing
  promise; neither did.
- **Sub-warp dots, lanes_per_dot=8 (contiguous 40-elem slices): rejected.**
  The a-fragment cvt:FFMA amortization drops 1:8→1:2 and activation loads
  duplicate 4×; dm2 (dots) @M=8 2.78→5.02 µs (+80%), while scatter+reduce
  only saves ~0.6 µs. Full: 5.86/8.00/21.63 at M=1/8/32.
- **lanes_per_dot=16 (interleaved k-slices): accepted, shipped.** Each dot
  is covered by 16 lanes (2 dot groups/warp, 4 rows/lane): SMEM staging
  32→16 cols, 5→4-stage reduce tree, cvt:FFMA 1:4, 2× activation-load
  duplication. dm2 @M=8 2.78→3.74 µs but scatter+reduce 3.27→2.18 µs.
  Ablate (no PDL): 4.51/4.51/4.64/6.30/8.67/14.85 vs 4.58/4.61/4.64/6.62/
  9.06/15.12 at LPD=32 — faster at every M, M=8 reaching chain parity there
  (6.30 vs 6.30). Official (PDL) bench: M ≤ 4 unchanged (4.64/4.64/4.67),
  M=8 6.62→6.56, M=16 9.02→8.86, M=32 14.91→14.88. Still bit-identical at
  M ≤ 2. LPD=8 under the same interleaved layout: 5.54/7.20/20.80 —
  still rejected. The lane split generalizes the LPD=32 layout (vec8
  k-groups {j + i·LPD} + a (64/LPD)-elem tail slice); `lanes_per_dot`
  remains a compile-time parameter for future sweeps.

Final-final FFMA pass (structural f32x2 + 256-bit loads + L1 no-allocate;
all behind compile-time flags `fma2` / `w256` / `wna`, defaults off):

- **Structural f32x2 (stream-pair vector<2> chains): rejected — FFMA2 does
  NOT dual-issue on SM103.** The DSL's loop vectorizer cannot express the
  pairing: `cutlass.range(..., vectorize=True)` on the dot loop fails to
  compile ("[vectorize=true] one dimension of the coordinates of
  cute.memref.load must be loop index" — it vectorizes loads/stores along
  the loop dim, so loop-carried scalar accumulators are ineligible). The
  working structural form keeps each stream pair as a TensorSSA vector<2>
  chain (`accp = accp + wr[rp, (None, v)].load() * ab[v]`, mul-first;
  products of bf16-origin operands are exact in fp32 so it stays
  bit-identical at M ≤ 2 — measured). SASS: 456 FFMA2 with proper
  `.F32x2.HI_LO` register pairs and only 16 MOVs in the whole kernel — the
  owner's packing-rearrangement hypothesis is thus tested cleanly (probe:
  `probe_f32x2_vec.py`). Result: dm2 (dots) @M=8 3.74→3.81 µs with HALF
  the FMA-pipe instructions — i.e. each FFMA2 occupies the pipe as long as
  2 scalar FFMAs; the FP32 rate is symmetric and there is no f32x2
  throughput doubling. Full kernel: M=1/4 regress 4.51/4.64→4.90/4.93;
  M=32 improves 14.80→13.74 µs (fewer instructions help the big-M
  latency mix) but that is outside the win window. Scalar FFMA stays.
- **256-bit weight/activation loads (`w256`, LDG.E.256): rejected.**
  Re-tiled the LPD=16 k-slice to a contiguous 32B main slice per lane
  (vec8 groups {2j, 2j+1} + vec4 tail) so autovec emits LDG.E.256 (SASS
  confirmed; requires a 32B alignment claim, i.e. `divisibility=16` at
  compile — the wrapper's divisibility=8 would cap it at 128-bit). dm0:
  4.54/4.54/4.67/6.27/8.67/14.75 vs 4.51/4.51/4.64/6.30/8.67/14.80 — a
  wash (±0.03 µs); dm2 @M=8 3.71 vs 3.74. The kernel is latency-bound,
  not LDG-count-bound. The contiguous split WITHOUT 256-bit (div=8)
  regresses (4.70/6.62) — interleaved slices coalesce better. xn epilogue
  loads can't be widened (each lane owns a scattered column). Not shipped:
  no gain, and it would need the wrapper's divisibility bumped.
- **L1 no-allocate on the weight stream (`wna`,
  `l1c_evict_priority=NO_ALLOCATE` via `cute.nvgpu.CacheEvictionPriority` —
  note the `cute.nvgpu` enum, the `cute.` one lacks `_to_ir`): rejected.**
  4.48/6.30/14.94 at M=1/8/32 — noise-level everywhere, including stacked
  with w256. Use-once weights were not evicting anything that matters
  (ncu already showed latency, not L1-capacity, dominance).

With both owner-directed experiments negative, the scalar-FFMA design stays
as shipped (CW=2, MB=4, m_splits=4, lanes_per_dot=16). **The FFMA design is
closed**: at M=8 the official gap to the chain is 3% (6.56 vs 6.34 µs) and
every lever in the FFMA design space has now been measured.

### Limiter at M ≥ 8 (measured on the pre-LPD16 config @M=32)

ncu: 168 regs/thread; block limits = 3 CTAs by registers AND by SMEM
(33.8KB staging); achieved 11.2/64 warps/SM (17.6%); stalls per issued
instruction: long-scoreboard 2.04 (cold w stream + read-once xn), wait 1.30
(fixed-latency FMA/LDS chains), short-scoreboard 0.25, mio_throttle 0.08 —
the SMEM pipe is NOT throughput-bound; everything is latency exposure. The
reduce+epilogue is a per-block serial chain (32 STS → 32 LDS → 31-FADD tree
→ sigmoid → 4 gather SHFL → guarded store) that the one-block-behind
software pipeline cannot hide at 3 CTAs/SM: at M=32 full−dots = 10.1 µs
over 2 blocks/CTA. The dots phase at 4.8 µs is within 1.5× of the
3.3 µs FFMA-issue floor (which does NOT halve via f32x2 on SM103 — see
above), so even perfect overlap would land ~5 µs at M=32
vs the 7.0 µs chain — but the reduce+epilogue would have to shrink ~3×,
which this FFMA+SMEM structure cannot do. (At M=8 the pipeline degenerates
to one block per CTA, so the chain is fully exposed: full−dots ≈ 4 µs.)

**FFMA design closed (final-final pass).** lanes_per_dot=16 narrowed M=8 to
0.97× of the chain (6.56 vs 6.34 µs) but did not clear it: the dots phase
grew (2.78→3.74 µs, cvt:FFMA 1:8→1:4) while scatter+reduce shrank
(3.27→2.18 µs) — the two are now near-balanced, and every FFMA-space lever
has been measured: regcap/CH/CW/MB sweeps, FFMA2 (explicit intrinsic AND
structural TensorSSA pairing — no dual-issue on SM103), LDS.128, butterfly
reduce, cp.async staging, grid reorder, .cg loads, LPD=8, 256-bit loads,
L1 no-allocate. Further M ≥ 8 gains require the tensor-core path (v4/v5
below); the scalar-FFMA design is closed with lanes_per_dot=16 as the
shipped configuration.

### mma.sync / tcgen05 status

The pre-pass recommendation stands and is now being pursued separately:
`_up_gate_mix_v4.py` (mma.sync) and `_up_gate_mix_v5.py` (tcgen05) exist in
`fused_up/` and are wired into the wrapper (`impl="v4"`,
`up_gate_mix_gemm_v4`, `UpGateMixV5`). M ≥ 8 fusion will be decided there,
not in the FFMA design.

Reproduce:

```bash
cd /home/thien/vllm/main && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python workspace/qwen4_exp_mhc_llgemm/bench_fused_up_v3.py
```

Raw data: `results_fused_up_v3.json`; sweep driver: `bench_v3_ablate.py
M regcap ch minb cw mb msplit lpd fma2 w256 wna div` (debug_mode 0=full,
1=no-epilogue, 2=loads+dot); correctness: `check_fused_up_v3.py` (wrapper
path), `check_v3_cw.py cw mb msplit lpd` (config sweep), and
`check_v3_flags.py lpd fma2 w256 wna div` (flag variants).

## v4: mma.sync m16n8k16 with swapped operands (`_up_gate_mix_v4.py`)

Final config: 160-CTA grid (16 output channels per CTA), 4 warps/CTA =
**one warp per hyper-connection stream**; tile_k=64, num_stages=3, ksplit=1.
The MMA operands are swapped vs a plain GEMM: output channels sit on MMA-M
(16) and tokens on MMA-N (8), so each `mma.sync.m16n8k16` computes
C[16 ch, 8 tok] = W_tile @ A_tile^T; the channel side packs densely (no
weight bandwidth wasted at M=1) while the token side pads. Key structures:

- **Per-warp weight pipeline**: each warp cp.async-streams its own stream's
  16ch x tile_k weight tiles through 3 SMEM stages; stages 0..ST-2 are issued
  BEFORE the PDL wait (persistent weight). Wait schedule is exact at both
  ends: `wait_group(min(ST-1, NUM_KT-1-kt))` at stage kt (a constant ST-2
  wait is over-strict mid-loop and racy on the tail).
- **All NT token tiles as one MMA-N dimension**: a single B fragment per
  k16 block covers every token tile (one ldmatrix copy + one gemm call per
  k-block instead of NT serialized copy->MMA pairs — measured 4x fewer copy
  instructions; note the DSL miscompiles *batched* ldmatrix copies into
  separate fragment slots, only adjacent copy->gemm pairs are safe).
- **Activations + xn** cooperatively cp.async-staged once per CTA (128
  threads, M-predicated rows) after the PDL wait; xn comes in as 32B-segment
  gathers (scalar 2B gmem gathers in the epilogue measured +5 µs at M=1).
- **Exchange + epilogue**: warps deposit fp32 gates in a +1-padded SMEM
  exchange, then 128 threads (one per (channel, token)) do bf16-round ->
  fastmath sigmoid -> x xn -> stream-serial sum -> x1/4 -> bf16 store, all
  SMEM loads batched before the math. The gate never touches gmem.
- M dynamic (two compiled variants, max_m 32/64), static grid, CUDA-graph
  safe, PDL preserved.

**Byte-domain swizzle gotcha**: CuTe DSL `make_swizzle` takes BYTE-domain
parameters (unlike element-domain C++ CuTe); for 16B copy/ldmatrix atoms the
conflict-free choice is `make_swizzle(log2(row_bytes)-4, 4, 3)` — i.e. 128B
rows -> S<3,4,3>, 64B rows -> S<2,4,3>. An element-domain formula silently
drops the swizzle for 64B rows, which cost 4-way ldmatrix bank conflicts on
the weight fragments (1.6M excessive SMEM wavefronts; kernel was 15.4 µs
before the fix, 5.6 µs after). `_ll_bf16_splitk.py::_make_smem_layout_AB`
appears to have the same mis-scaling — flagged, not modified.

### Correctness (per M)

**Bit-identical to the production chain at M <= 48** (maxdiff 0.0); M=64
shows one borderline bf16 rounding flip (maxdiff 9.8e-4, 1 ulp — the
fp32-partial sigmoid path rounds a razor-edge element differently). maxdiff
vs fp32 reference <= 0.0039 at all M (bf16 output quantum). `check_v4.py
tile_k num_stages debug_mode ksplit`.

### Timing (µs), fused v4 vs unfused chains (cold-L2 CUPTI CUDA graph)

| M   | best unfused chain | nvjet GEMM alone | fused v4 | speedup vs chain | v3 (FFMA) |
|-----|--------------------|------------------|----------|------------------|-----------|
| 1   | 6.18 (cublas)   | 4.51 | **5.54** | **1.12x** | 4.64 |
| 2   | 6.24 (cublasLt) | 4.48 | **5.54** | **1.13x** | 4.64 |
| 4   | 6.30 (cublas)   | 4.54 | **5.57** | **1.13x** | 4.67 |
| 8   | 6.34 (cublas)   | 4.48 | **5.66** | **1.12x** | 6.56 |
| 16  | 6.85 (cublas)   | 4.86 | **6.02** | **1.12x** | 8.86 |
| 32  | 7.04 (cublas)   | 4.86 | **6.82** | **1.03x** | 14.88 |
| 48  | 8.74 (cublas)   | 4.70 | 9.06  | 0.96x | 20.93 |
| 64  | 8.90 (cublas)   | 4.74 | 9.92  | 0.90x | 26.66 |

**Verdict: v4 beats the unfused chain at every M <= 32** (1.03-1.13x) and
dominates v3 for M >= 8. v3 remains the best fused option at M <= 4 (4.64
vs 5.54 µs — its register-resident weights skip the SMEM pipeline entirely),
so the wrapper should dispatch v3 for M <= 4 and v4 for 8 <= M <= 32. At
M >= 48 the chain wins (NT=8 token-tile work doubles; not tuned further —
stretch case).

### Limiter analysis

Phase breakdown (debug modes, tk64-st3, M=1/8/32): weight-streaming only
4.38/4.58/4.74 µs (debug 3) — already at/beating nvjet's effective cold-DRAM
rate (nvjet shows the same ~1.4 TB/s on this 6.5 MB read-once stream, so
~4.5 µs is the pattern floor and the original ~2-3 µs goal is not reachable
under cold-L2); +ldmatrix/MMA 5.15/5.22/5.44 (debug 2); +exchange 5.22/
5.31/5.57 (debug 1); +epilogue 5.54/5.70/6.82 (debug 0). The remaining
non-streaming cost (~1.2 µs at M=1, ~2.1 µs at M=32) is per-warp latency
exposure at ~1 warp/scheduler (160 CTAs = 640 warps on 148 SMs), not DRAM
or SMEM throughput. Things measured neutral-or-worse: deeper pipelines
(st 4/5/8/10), ksplit=2/4/5 (8-20 warps/CTA), min_blocks_per_mp=2, .cs
evict-first cache mode, dynamic M-based NT skipping (branch overhead),
batched B-fragment preloads (DSL miscompiles them). Next lever if M=32/64
ever matters: TMA bulk-tensor streaming + mbarrier pipeline to raise
aggregate in-flight bytes per SM.

Reproduce:

```bash
cd /home/thien/vllm/main && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
  .venv/bin/python workspace/qwen4_exp_mhc_llgemm/bench_fused_up_v4.py
# config via env: V4_TILE_K=64 V4_NUM_STAGES=3 V4_KSPLIT=1
```

Raw data: `results_fused_up_v4.json`; sweep driver: `bench_v4_sweep.py`;
phase breakdown: `bench_v4_debug.py tile_k num_stages` (debug_mode 3=loads
only, 2=+MMA, 1=+gate store, 0=full).

## v5: tcgen05 with swapped operands (`_up_gate_mix_v5.py`)

tcgen05.mma (tensor-memory accumulators) instead of mma.sync, with the GEMM
operands **swapped**: output channels on the tcgen05-M dim, tokens on the
tcgen05-N dim. Grid = (40, ceil(M/mma_n)) CTAs — 40 channel tiles of 64
channels (each CTA computes all 4 streams for its channels) x token tiles
(M > mma_n spills to grid-y; the weight stream is shared across concurrent
tile CTAs via L2). Four tmem accumulators (one per stream s), so the
sigmoid-mix epilogue is purely per-lane after `tcgen05.ld` — the [M, N]
gate is never materialized.

Warp specialization (flashinfer low-latency tcgen05 idiom): warp 0 = weight
TMA producer through a 16-deep stage ring, launched **before**
`griddepcontrol_wait` (weights are producer-independent, so the cold-DRAM
stream overlaps the predecessor kernel); warp 1 = activation TMA (after the
PDL wait); warp 2 = MMA (owns tmem alloc/dealloc); warps 4-7 (+ 8-11) =
epilogue. The epilogue prefetches its `xn` elements into registers *before*
waiting on the MMAs (independent of the gates), then tcgen05.ld's the 4
accumulators with an identical (channel, token)->lane mapping, applies the
production rounding boundary (fp32 gate -> bf16 -> fp32 -> sigmoid), mixes
s-ordered (FFMA-fused), scales 1/HC, stores bf16.

Final config: `mma_m=64, bk=64, num_w_stage=16`, per-M token bucket
`mma_n in {8, 16, 32, 64}` (one compilation per bucket, M dynamic within a
bucket, CUDA-graph safe), `sigmoid_mode="tanh"`, per-bucket epilogue
warpgroups `{8:1, 16:2, 32:2, 64:4}` (extra warpgroups split the token
columns; tcgen05.ld lane groups repeat per warpgroup, so all read the same
tmem rows at different column offsets). Large-M bucket rule (swept): M<=64
single tile; 64<M<=96 -> mma_n=32 (3 tiles = 120 CTAs, one wave);
M>96 -> mma_n=64 (ceil(M/64) tiles, one wave up to M=192). Hard limits:
M <= 512, mma_n <= 128 (4 streams x mma_n tmem cols <= 512); the mma_n=128
bucket exists but is never auto-selected (heavier per-CTA epilogue even at
nwg=4 — loses to tiled mma_n=32/64 at every M).

### Correctness (per M)

- `sigmoid_mode="tanh"` (default, fastest): maxdiff vs production chain
  <= 0.0078 (1 bf16 ulp) on ~1% of elements, from `tanh.approx.f32`'s
  ~5e-4 abs error on the sigmoid. maxdiff vs the fp32 reference is
  <= 0.0039 at M <= 64 (identical to the bit-exact mode — the tanh error
  is below the bf16 output quantum) and 0.0071 at M >= 65 (same for both
  modes; more elements -> more extreme values). Not bit-identical to the
  chain.
- `sigmoid_mode="exact"` (exp+rcp, 2 MUFU per sigmoid): **bit-identical to
  the production chain at M <= 16**, 1e-6 at M=32, ~1e-3 at M=64..128,
  ~2e-3 at M >= 192 (same borderline bf16 rounding flips as v4).
- Token-tile path (M > mma_n) verified at M in {65, 96, 128, 129, 192, 256,
  384, 512} (`probe_v5_largem.py`).

### Timing (us), fused v5 vs unfused chains (cold-L2 CUPTI CUDA graph)

| M   | best unfused chain | cuBLAS GEMM alone | fused v5 (tanh) | fused v5 (exact) | speedup vs chain (tanh) |
|-----|--------------------|-------------------|-----------------|------------------|-------------------------|
| 1   | 6.34 (cublas)   | 4.54 | **5.58** | 5.95  | **1.13x** |
| 2   | 6.40 (cublas)   | 4.54 | **5.60** | 6.30  | **1.14x** |
| 4   | 6.37 (cublas)   | 4.58 | **5.60** | 6.30  | **1.14x** |
| 8   | 6.37 (cublasLt) | 4.51 | **5.57** | 6.27  | **1.14x** |
| 16  | 6.91 (cublasLt) | 4.86 | **5.82** | 6.50  | **1.19x** |
| 32  | 7.07 (cublas)   | 4.90 | **6.24** | 7.90  | **1.13x** |
| 48  | 8.83 (cublas)   | 4.67 | **7.07** | 8.67  | **1.25x** |
| 64  | 8.99 (cublasLt) | 4.77 | **7.30** | 8.90  | **1.23x** |

**Verdict: v5-tanh beats the unfused chain at every M in {1..64}**
(1.13-1.25x), the only fused variant that also wins at M=48/64. It sits
within ~1.0 us of the bare cuBLAS GEMM at M <= 16. Run-to-run spread of the
harness is ~+-0.3-0.5 us between processes (medians within a process are
stable to ~0.05 us).

### Large M (65..512): dispatch crossover

Token tiling (grid-y) extends the kernel to M <= 512. Per-M table
(two runs agreed to within 0.05 us; run 1 shown):

| M   | best unfused chain | GEMM alone | mix alone | fused v5 (tanh) | fused v5 (exact) | speedup (tanh) |
|-----|--------------------|------------|-----------|-----------------|------------------|----------------|
| 64  | 9.06  | 4.83 | 2.98 | **7.52**  | 9.22  | **1.20x** |
| 96  | 9.47  | 5.18 | 3.33 | **6.56**  | 8.16  | **1.44x** |
| 128 | 9.47  | 5.15 | 3.62 | **7.71**  | 9.31  | **1.23x** |
| 160 | 9.98  | 5.54 | 3.94 | **7.81**  | 9.38  | **1.28x** |
| 192 | 10.08 | 5.82 | 4.26 | **7.90**  | 9.50  | **1.28x** |
| 224 | 10.18 | 5.79 | 4.67 | 12.00 | 15.14 | 0.85x |
| 256 | 10.37 | 5.79 | 4.83 | 12.38 | 15.49 | 0.84x |
| 384 | 11.71 | 6.94 | 5.79 | 13.12 | 16.13 | 0.89x |
| 512 | 12.06 | 6.98 | 6.75 | 17.57 | 22.27 | 0.69x |

**Crossover: v5-tanh wins through M=192 and loses from M=224 on.**
Recommended production dispatch gate: **M <= 192 -> fused v5-tanh;
M > 192 -> unfused chain**. (v5-exact also wins through M=192, barely.)
The cliff at M=224 is CTA-wave quantization, not traffic: one wave =
152 CTAs = 40 channel tiles x <=3 token tiles, so M <= 192 runs in a single
wave (~6.5-7.9 us flat — more CTAs per wave actually *helps*: M=96/128 beat
M=64 because 80-120 weight-TMA streams hide DRAM latency better than 40);
M=224 needs 4 token tiles = 160 CTAs -> 2 waves -> ~2x. The chain's mix
kernel scales ~linearly (~45 KB/token: 3.0 -> 6.8 us at M=64 -> 512) while
its GEMM stays flat, so the chain degrades gracefully past the crossover.

Raw data: `results_fused_up_v5_largem.json`; driver:
`bench_fused_up_v5_largem.py`; config sweep: `bench_v5_sweep_largem.py`.

### Limiter analysis

- **Cold-L2 weight-read floor = ~4.5-4.7 us.** The 6.5 MB read-once weight
  stream measures ~1.4 TB/s regardless of CTA count (Triton pure-read kernel:
  4.70 us at 400 CTAs, 4.74 at 800, 5.54 at 1600; cublasLt/nvjet GEMM:
  4.58 us; this kernel's TMA-only path: 5.0-5.3 us). This is a system floor
  of the cold-L2 methodology, NOT CTA-parallelism-limited — so the feared
  40-CTA grid is not the binding constraint, and cluster/k-split/stream-split
  rewrites for more CTAs would not help. It also means a ~2-3 us fused kernel
  is physically impossible under cold-L2 (weights alone cost 4.5 us); only
  warm-L2 production conditions (weights resident) or PDL overlap change that
  calculus.
- **Sigmoid MUFU was the epilogue wall.** Epilogue cost over MMA+TMA-only
  (debug modes) at M=1/32/64: exact = +1.2/+4.2/+13.7 us; tanh (1 MUFU per
  sigmoid, verified `tanh.approx.f32` -> `MUFU.TANH` in PTX/SASS) =
  +0.3/+1.1/+4.1 us; tanh + 2nd epilogue warpgroup ~= +0.3/+0.5/+1.6 us.
  ncu showed the MUFU(xu) pipe at only ~4% — latency/ILP-bound with 4 warps,
  ~2 MUFU/cyc/SM effective.
- **Weight ring depth matters:** `num_w_stage` 8 -> 16 took M=1 from 6.35 to
  5.60 us (deeper ring hides more DRAM latency before the MMA drain);
  20/24 are flat-to-worse. `bk=64 > 32`; `mma_m=128` does not compile
  (SMEM over capacity at sw=16).
- **Large M is CTA-wave-quantized, not traffic-bound.** One wave = 152 CTAs;
  any (mma_n, tiles) with 40*tiles <= 152 lands at ~6.5-7.9 us, and fuller
  waves are *faster* (more concurrent weight-TMA streams). Crossing into 2
  waves ~doubles latency (M=192: 7.90 -> M=224: 12.00). The mma_n=128
  bucket loses to tiled mma_n=32/64 at every M despite fewer CTAs — its
  per-CTA epilogue is too heavy even at nwg=4 (VALS=16/thread/stream, but
  512 tmem cols + 80 KB activation SMEM), and nwg=2 spills registers
  (M=96: 16.4 us vs 12.9 at nwg=4 vs 7.6-8.1 tiled).
- **PDL on/off: no measurable difference in this harness** (kept on for
  production, where the predecessor-overlap is real).
- Remaining gap to the 4.5-4.7 us floor at M=1: ~0.9-1.1 us of MMA-pipeline
    - tmem + epilogue latency on the tail of the weight stream.

### v5 (tcgen05) vs v4 (mma.sync)

Owner decision: **v5-tanh is the production candidate; v4 is dropped**
(non-SM100 portability of tcgen05 does not matter for this deployment).
Kept for the record:

| M   | v4 (bit-identical) | v5 tanh (1-ulp, ~1%) | v5 exact (bit-identical) |
|-----|--------------------|----------------------|--------------------------|
| 1   | 5.54 | **5.58** | 5.95  |
| 8   | 5.66 | **5.57** | 6.27  |
| 16  | 6.02 | **5.82** | 6.50  |
| 32  | 6.82 | **6.21** | 7.90  |
| 48  | 9.06 | **7.42** | 10.27 |
| 64  | 9.92 | **7.78** | 10.56 |

- **v5-tanh is the best fused kernel at every M** (ties v4 at M <= 8, wins
  M >= 16, and is the only fused kernel that beats the chain at M >= 48).
  tcgen05's tmem accumulators keep the epilogue register-flat as M grows,
  where v4's mma.sync accumulators spill into per-warp latency exposure.
- v4's only remaining edge is strict bit-identity with the production chain
  (at M <= 48); v5-exact is bit-identical only to M <= 16. If bit-identity
  ever becomes a hard requirement, that is the fallback — otherwise ship
  v5-tanh gated at M <= 192.

Reproduce:

```bash
cd /home/thien/vllm/main && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 \
  .venv/bin/python workspace/qwen4_exp_mhc_llgemm/bench_fused_up_v5.py
# large M: bench_fused_up_v5_largem.py (results_fused_up_v5_largem.json)
# config knobs: fused_up/up_gate_mix.py UpGateMixV5(mma_m, bk, num_w_stage,
#   sigmoid_mode, num_epilog_wg, mma_n_override); sweeps:
#   bench_v5_sweep2.py {sw,bk}, bench_v5_sweep_largem.py <M-list>;
#   phase breakdown: bench_v5_ablate.py <M-list>
```

Raw data: `results_fused_up_v5.json`.

## v5 PDL HC-chain: init-hiding under the down projection

Question: in production, v5 runs immediately after `hc_down_silu` with a
real dependency edge (`lora = down_out[:, :320]`, strided row-stride 336 for
M <= 48). How much of v5's ~5.6-7.4us can PDL hide under the down op, and
what does the fused boundary save end-to-end vs the production up chain?

Setup (`bench_v5_hc_chain.py`, GB300, cold-L2 CUPTI CUDA graph, 200 iters,
median; two independent runs, spread <= 0.06us on chain totals):

- chainA = `[down -> v5]` with v5 PDL on / off
- chainB = `[down -> F.linear + hc_gate_mix]` (production up, as-is)
- down = production `hc_down_silu` (fused cute-dsl kernel, M <= 48) or
  `F.linear -> hc_silu` (cublas + splitKreduce + Triton silu, M > 48)
- correctness: v5 vs production chain maxdiff <= 0.0039 at every M

| M  | v5 alone PDL-off | v5 alone strided-a | chainA: v5 dur@start (PDL on) | chainA PDL on | chainA PDL off | chainB prod | B - A |
|----|------------------|--------------------|-------------------------------|---------------|----------------|-------------|-------|
| 1  | 5.63 | 5.63 | 5.28 @ 3.84 | 9.15  | 9.34  | 9.82  | 0.67 |
| 2  | 5.60 | 5.60 | 5.38 @ 3.87 | 9.25  | 9.89  | 9.73  | 0.48 |
| 4  | 5.50 | 5.50 | 5.47 @ 5.38 | 10.85 | 11.07 | 11.46 | 0.61 |
| 8  | 5.60 | 5.60 | 5.79 @ 5.09 | 10.88 | 11.15 | 11.01 | 0.13 |
| 16 | 5.89 | 5.89 | 5.70 @ 5.15 | 10.85 | 11.10 | 11.68 | 0.83 |
| 32 | 6.14 | 6.21 | 6.05 @ 5.79 | 11.87 | 11.89 | 12.16 | 0.29 |
| 48 | 7.23 | 7.33 | 7.14 @ 6.46 | 13.60 | 13.98 | 14.75 | 1.15 |
| 64 | 7.42 | 7.49 | 6.62 @ 9.28 | 15.94 | 16.29 | 17.60 | 1.66 |
| 96 | 6.66 | 6.66 | 5.76 @ 11.46| 17.25 | 17.63 | 20.29 | 3.04 |

(times in us; `@` = median start offset from iteration t0)

Findings:

- **M <= 48: PDL hides the launch gap only, not the v5 body.** The down
  kernel's `griddepcontrol.launch_dependents` fires at kernel end, so v5
  starts within ~0.15us of down's end either way; PDL-off pays a
  0.45-0.75us launch gap (v5 start 4.32 vs 3.84 at M=1). Net chain gain
  from PDL: 0.0-0.64us (avg ~0.3us) — the raw gap minus a small PDL-on
  duration increase (the griddepcontrol wait is inside the kernel).
- **M > 48: genuine overlap, but small.** Down becomes
  cublas + splitKreduce + Triton `hc_silu` (silu itself launches with PDL).
  v5 starts 0.15-0.2us *before* silu ends (M=64: start 9.28 vs silu end
  ~9.44) — the only real body-hiding in the chain. Net gain 0.35-0.38us.
  PDL is not more valuable at M > 48 despite the 3-kernel down.
- **Fused boundary saving (chainB - chainA) grows with M**: 0.1-0.8us for
  M <= 32, 1.15us @48, 1.66us @64, 3.04us @96. At M=8 the two nearly tie
  (chainB's cublas+gate_mix happens to be fast there). Fused wins at
  every M.
- **Strided-a is free**: stride-336 `a` costs <= 0.1us standalone at any M
  (largest +0.10us @48). No contiguous copy needed — v5 consumes the down
  output slice directly via the strided `mark_layout_dynamic` path.
- Anomaly worth knowing: the production down kernel is *slower* at M=4
  (5.50us) and M=8 (4.96us) than at M=1-2 (3.9us) — visible in both chainA
  and chainB, independent of v5.

Reproduce:

```bash
cd /home/thien/vllm/main && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=2 \
  .venv/bin/python workspace/qwen4_exp_mhc_llgemm/bench_v5_hc_chain.py
```

Raw data: `results_v5_hc_chain.json` (per-kernel CUPTI durations and start
offsets for all five configs).
