# mHC hc-up Fusion — Tracking

Prototype workspace for fusing the qwen4_exp mHC **up** GEMM (`[M,320] @ [10240,320]^T`)
with the sigmoid gate-mix epilogue (4 streams × 2560 → `out[M,2560]`).

Down-side fusion (`hc_down_silu`) is **done and merged upstream** (PR #58957). This file
tracks only the up-side work.

- Hardware: GB300 (SM103), cold-L2 CUDA-graph CUPTI medians, µs.
- Weight stream: 6.5 MB read-once, floors at ~4.4–4.7 µs cold (~1.4 TB/s), invariant to CTA count.
- Inter-run spread: ±0.3–0.5 µs.
- No PDL in the standalone numbers; PDL is on/off explicitly in the chain table.

## Candidates

| candidate | what it is | status |
|---|---|---|
| baseline | current production: cuBLAS (nvjet) GEMM + `hc_gate_mix` | — |
| v3 FMA | FMA kernel, f32x2, strided-`a` supported | small-M candidate — wins M≤4, loses from M=8; FFMA2 does not dual-issue on SM103 (SASS-proven) |
| v4 mma.sync | mma.sync m16n8k16, swapped operands | **dropped** — we don't care about non-SM100; numbers in `RESULTS_FUSED_UP.md` |
| v5 tcgen05 | flashinfer experimental API, MMA_M=64 (4 stream accs), num_w_stage=16, epilogue warpgroups {8:1,16:2,32:2,64:4}, token-tiled grid-y | large-M candidate (96–384), ≤1 bf16 ulp on ~1% elems |
| v6 tcgen05 | raw standard-API rewrite of v5 (no tiled_mma / experimental ext) | reference only — GEMM ≥ v5, fused behind v5; superseded by v7 |
| v7 tcgen05 | MMA_M=128: all 4 streams × 32 ch in ONE accumulator (streams share x, packing is free); grid = 80 CTAs; K=320 = exactly 5 stages × BK=64 (ring never wraps → no release/phase toggles, fully unrolled); 4 warps (8/16 in the fused epilogue for MMA_N≥64/96); 4D weight TMA (8-row stream-interleave → intra-warp butterfly mix); xn TMA→smem during mainloop, preloaded to regs before the MMA-drain wait; all warps wait the commit barrier; TMA store | **best M=2–64** (chain); bit-identical to v6/cuBLAS |

## Standalone kernel (µs, lower is better)

`nvjet GEMM` = `F.linear` alone — the optimization floor. `gate-mix` = `hc_gate_mix`
alone — the removable part. (Their sum slightly exceeds the chained baseline because the
baseline already overlaps gate-mix with the GEMM tail via PDL.)

| M | baseline | nvjet GEMM | gate-mix | v3 FMA | v5 tcgen05 | v7 tcgen05 | best fused | speedup |
|---|---|---|---|---|---|---|---|---|
| 1   | 6.34 | 4.54 | 2.21 | **4.64** | 5.58 | 4.77 | v3 4.64 | 1.37× |
| 2   | 6.40 | 4.58 | 2.18 | **4.64** | 5.60 | 4.77 | v3 4.64 | 1.38× |
| 4   | 6.37 | 4.54 | 2.24 | **4.67** | 5.60 | 4.83 | v3 4.67 | 1.36× |
| 8   | 6.37 | 4.58 | 2.30 | 6.56 | 5.57 | **4.83** | v7 4.83 | 1.32× |
| 16  | 6.91 | 4.86 | 2.50 | 8.86 | 5.82 | **5.06** | v7 5.06 | 1.37× |
| 32  | 7.07 | 4.93 | 2.56 | 14.88 | 6.21 | **5.44** | v7 5.44 | 1.30× |
| 48  | 8.86 | 4.86 | 2.82 | — | 7.07 | **6.05** | v7 6.05 | 1.46× |
| 64  | 9.06 | 4.77 | 2.98 | — | 7.30 | **6.11** | v7 6.11 | 1.48× |
| 96  | 9.47 | 5.22 | 3.33 | — | **6.56** | 6.88 | v5 6.56 | 1.44× |
| 128 | 9.47 | 5.12 | 3.65 | — | 7.71 | **7.68** | v5/v7 ≈ tie | 1.23× |
| 160 | 9.98 | 5.57 | 4.00 | — | **7.81** | ✗ | v5 7.81 | 1.28× |
| 192 | 10.08 | 5.82 | 4.16 | — | **7.90** | ✗ | v5 7.90 | 1.28× |
| 224 | 10.18 | 5.82 | 4.67 | — | 10.53 (mn96) | ✗ | v5 ≈ tie | ~1.0× |
| 256 | 10.37 | 5.86 | 4.74 | — | 10.47 (mn96) | ✗ | v5 ≈ tie | ~1.0× |
| 384 | 11.71 | 6.85 | 5.89 | — | 11.65 (mp2) | ✗ | v5 ≈ tie | ~1.0× |
| 512 | 12.06 | 7.01 | 6.85 | — | 15.49 (mp3) ✗ | ✗ | baseline | 1.00× |

Notes:

- v7 column: `probe_v7_fused.py` (±0.3–0.5 µs spread). With the EPI4/EPI8
  epilogue splits + xn-preload-before-drain + all-warps drain wait, v7 wins
  M=8–64 outright, ties 128, and is within noise of v5 at 96.
- v7 GEMM-only (TMA-store epilogue scaffold, bit-identical to nvjet):
  4.67/4.77/4.86/5.12/5.44/5.57/5.95/6.50 at M=1/8/16/32/48/64/96/128 —
  within +0.2 of nvjet at M≤32; the residual at M≥48 is the scaffold's
  serial tmem→smem staging (bank-conflicted across stream quarters; padding
  can't fix it — TMA box bases need 128B alignment = the bank period).
  The base WITHOUT epilogue (no_epi) is 4.48–4.77 ≈ nvjet everywhere.
- v7 not run past 128: M>128 needs grid.y≥2 → >148 CTAs → 2-wave cliff (measured:
  mma_n=64 × grid.y=2 at M=128 = 11.20 vs 8.90 single-tile). Fused smem caps
  mma_n at 128 (xn tile 128·mma_n·2 B would overflow 227 KB at 192).
- v5 loses from M=224: CTA-wave cliff (one wave = 152 CTAs = 40 channel tiles × ≤3 token tiles).
- v5-tanh error vs fp32-ref identical to exact mode (≤0.0039); v5-exact is bit-identical M≤16.
- v3 FMA is bit-identical at M≤2. v7 is bit-identical to v6 at every M (md=0.0000).
- Gap to the GEMM floor: v3 within ~0.1 µs of nvjet at M≤4 (fusion essentially free);
  v7's gap ~0.1–0.6 µs at M=8–64; v5's gap ~1.0–1.3 µs at M=8–192. The absorbed
  gate-mix costs 2.2–4.2 µs standalone.

**Recommended dispatch:** v3 FMA M=1, **v7 M=2–64** (solid standalone + chain),
v5 M=96 (still ahead in-chain: 17.31 vs 18.08), v5/v7 at 128 (v5 marginally
ahead in-chain: 17.41 vs 17.79), v5 M=129–192 (stock buckets), cuBLAS fallback
above 192 (mn96 / multi-pass variants tied the baseline there at best — large-M
is not the target; history in `RESULTS_FUSED_UP.md`). v3 takes the strided
lora slice directly.

## mHC chain with PDL overlap — `[down → up → downstream]` (µs, one CUDA graph)

Full 3-kernel boundary around the up kernel: **before** = `hc_down_silu`
(early-PDL variant; self-dispatches to cublas+splitk+silu above M=48),
**after** = `F.linear(blk, [2048,2560] bf16)` standing in for the
attention-qkv / MLP gate_up GEMM that consumes `block_input`. PDL on
everywhere (production setting). `bench_hc_chain_ds.py`,
`results_hc_chain_ds.json`.

| M | baseline | v3 | v5 | v7 |
|---|---|---|---|---|
| 1  | 14.21 | **12.80** | 13.20 | 13.78 |
| 2  | 14.59 | 13.70 | 13.89 | **12.88** |
| 4  | 15.71 | 14.24 | 14.53 | **13.92** |
| 8  | 16.19 | 16.83 ✗ | 14.91 | **14.18** |
| 16 | 16.70 | 18.94 ✗ | 15.36 | **14.30** |
| 32 | 17.86 | — | 15.71 | **15.10** |
| 48 | 20.13 | — | 17.73 | **16.42** |
| 64 | 22.78 | — | 20.86 | **19.58** |

(v3 benched at M≤16 only — small-M config, max_m=32. The margin over the
baseline grows with M because standalone `gate_mix` cost grows — 3.3 µs @M=1 →
5.5 µs @M=48 — while the fused epilogue stays near-free. v3 at M=8 regresses in
the overlap regime, matching the 2-kernel chains. 2-kernel `[down → up]`
history: `results_hc_chain_pdl.json`, where v5 keeps M=96/128 — 17.31 vs v7
18.08, 17.41 vs 17.79.)

**PDL overlap, upstream edge (`down → up`)**: the down kernel's early
`launch_dependents` lets the up kernel launch ~1.3–1.9 µs into the chain at
M=8–48. Attribution probes (`bench_hc_chain_pdl.py`: v7_nopdl / wait-before-TMA
/ early): early-launch overlap (launch + mbarrier init + tmem alloc +
descriptor prefetch) = 0.8–1.2 µs at M≤48; weight pre-streaming under the down
kernel (TMA ring issues before `griddepcontrol_wait`) = 0.2–0.6 µs. v7 with PDL
off ≈ unfused baseline at M≤8 — the chain win is roughly half PDL, half fusion.

**PDL overlap, downstream edge (`up → ds`)**: the ds GEMM PDL-launches as soon
as the up kernel fires `launch_dependents`. v7 fires from its TMA warp right
after issuing TMAs, so ds starts 1.7–2.7 µs BEFORE v7 ends (M=1: ds @4.64, v7
ends ~7.2) and all of ds's launch+startup hides under the mainloop — the
exposed ds cost ≈ its compute only. The overlap is one-directional: ds
consumes `block_input`, so it must wait for the up kernel's stores — our
drain+epilogue tail is NOT hidden by downstream (inherent to
producer→consumer).

Why v7@M=2 ≈ v3@M=1 in this table (anomaly, resolved): v3 fired
`launch_dependents` at kernel END, so ds's startup was fully exposed; hoisting
the fire to CTA start (kept — bit-identical) gained only ~0.2 µs at M=1 but
0.5–1.4 µs at M=8/16. The residual M≤4 gap is structural: v3's grid is
register-saturated (255 regs/thread), so ds's 256-thread CTAs cannot co-reside
and only launch as v3 retires — while v7's lean 80-CTA grid leaves SM room and
ds pre-launches deeply. The M=1→M=2 compute delta is ~0.1 µs, so v7's overlap
advantage can outweigh v3's standalone edge. The same early-fire hoist was
tried on v5 and REVERTED: ds CTAs launched early but contended with v5's
weight stream (worse at M≤4, wash at M≥32).

Why the v7 column shows M=2 (12.88) FASTER than M=1 (13.78) — debugged, not
ours: v7 is M-invariant (ends ~7.2 µs into the chain at both M; standalone
4.77 at both; `[v7 → ds]` alone: 10.98 vs 10.80 ≈ flat). The inversion
reproduces in `[down → ds]` WITHOUT v7 (11.55 vs 10.86). Drivers: (1) the down
FMA kernel bakes M into its compile key for M≤4 (`hc_down_silu.py:55`), so M=1
and M=2 are different binaries (90 vs 128 regs/thread) with different SM
retirement patterns; (2) the ds nvjet kernel
(`nvjet_sm103_tst_32x64_64x16_4x1_v_bz_TNN`) is a whole-SM kernel — 64 CTAs ×
(255 regs × 256 thr = full 64K register file, 205 KB smem) — so each CTA needs
a COMPLETELY empty SM and its completion is hostage to the predecessors'
SM-emptying pattern. Padding the M=1 chain's down kernel to the M=2 binary
recovers ~half the gap; the residual is the M=1 ds shape being ~0.3–0.5 µs
worse in the overlap regime (standalone ds is flat: 7.04/6.91/6.91 at
M=1/2/4). v3/v5/prod columns are normally ordered. Treat the v7 M=1/M=2
inversion as co-scheduling noise around the whole-SM stand-in consumer, not a
kernel property.

Caveat: timing-only at the ds boundary. Production correctness requires the
downstream GEMM to be PDL-aware (wait before reading `blk`) — the same
assumption existing vLLM PDL chains already make.

## e2e estimate (historical vigil A/B) — why this is stashed, not merged

Reference point: the merged down-silu fusion (PR #58957) gained ~1.4–3 µs/block
at kernel level and measured (vigil A/B, Qwen3.8-Flash-Next, TP4, 8k-in/1k-out
decode, `vigil_hc_down_silu_ab_latest.yaml`): TPOT 4.574→4.444 ms at c=1
(−130 µs/tok, −2.8%), 6.259→6.130 at c=4 (−2.1%), 10.082→9.681 at c=16
(−4.0%), parity at c=64. 130 µs / 48 mHC layers ≈ 2.7 µs/block — the
kernel→e2e scaling is linear in layer count.

This work (up fusion, chain win vs prod): 1.4 / 1.7 / 1.8 / 2.0 / 2.4 / 2.7 /
3.7 / 3.2 µs/block at M=1/2/4/8/16/32/48/64. ×48 layers ≈ 67–115 µs/token at
c=1–16 → **~1.2–1.6% decode TPOT** (c=1: ~67 µs on 4.44 ms ≈ 1.5%; c=16:
~115 µs on 9.68 ms ≈ 1.2%). Verdict: real but small; keep as prototype branch,
revisit if the up GEMM base gets closer to nvjet or more epilogue ops fold in.

## Stream-floor characterization (probe_stream_floor.py, results_stream_floor.json)

Raw read-once streaming, cold L2, CUPTI medians (µs / GB/s):

- Model: **~2.3 µs fixed startup** (launch + mbar init + descriptor prefetch +
  HBM latency) **+ ~5 TB/s marginal** cold; warm L2 marginal ~15 TB/s.
- **Fixed-size stream is FLAT in CTA count past ~80 CTAs**: 6.25 MB in 3.90 µs
  at G=80, 3.97 at G=160, 3.97 at G=222. → more-CTA weight tilings
  (BLOCK_N=80/96) are DEAD for the weight stream.
- In-flight depth: 5×16KB boxes ≈ 10×8KB (3.90 vs 4.00 at G=80); 4KB boxes
  worse. Deeper rings buy nothing.
- **TMA > LDG**: 6.25 MB at G=80: TMA 3.90 vs LDG.128 u16 4.64 (LDG u8 worse;
  160KB/CTA: TMA 5.06 vs LDG 8.51). LDG streaming is not competitive.

Consequences: v7's weight stream is already at the machine floor; the only
levers are (a) overlapping the fixed startup via PDL (already done), (b) the
epilogue tail (see below), (c) removing one kernel's fixed cost entirely
(fused down+up, phase 3).

## v7 phase decomposition (probe_v7_phases.py, gemm_only)

Extended probe set (all deltas vs `stream`, µs):

| mma_n | stream | commit_only | mma_nowait | serial_mma | late1 | late4 | no_epi | full |
|---|---|---|---|---|---|---|---|---|
| 32 | 3.78 | +0.19 | +0.19 | +0.93 | +0.54 | +0.58 | +0.70 | 4.99 |
| 64 | 3.84 | +0.19 | +0.19 | +1.02 | +0.48 | +0.61 | +0.77 | 5.50 |
| 96 | 3.84 | +0.19 | +0.19 | +1.12 | +0.54 | +0.64 | +0.83 | 5.98 |
| 128 | 3.87 | +0.22 | +0.20 | +1.22 | +0.58 | +0.69 | +0.90 | 6.46 |

- `commit_only` (alloc + commit an empty MMA group): **+0.19 µs** = the
  commit/mbarrier signal path floor.
- `mma_nowait` (all 20 MMAs issued immediately, garbage): also +0.19 — MMA
  execution hides entirely under the stream when issued early.
- `serial_mma` (wait all stages, then all MMAs): MMA throughput =
  0.74–1.25 µs for 20 MMAs (~64 cyc/instr at mma_n=32 = 2048 FLOP/cyc;
  sublinear in mma_n — narrow-N wastes the datapath but MMA is never the
  bottleneck).
- `late1` (ONE MMA issued after the full stream): **+0.5 µs** — the tcgen05
  pipeline-drain + commit latency for a minimal batch. This is the hardware
  floor of "GEMM base": any correct kernel pays it after the last byte
  lands. `no_epi` is only +0.16 above `late1` → the pipelined mainloop
  already hides everything hideable. **The GEMM base is AT the floor:**
  stream (3.78, HBM) + drain (~0.5, HW) + epilogue. nvjet pays the same
  (nvjet M=1 GEMM = 4.48 == no_epi mma_n=32).
- `no_fence` ≡ `no_epi`: the per-stage `tcgen05.fence_after_thread_sync`
  is free (and arguably unneeded — TMA→MMA ordering is via the mbarrier;
  kept anyway to match the in-tree/gn-kernels idiom).
- `spin_wait` (non-suspending `mbarrier.test_wait` spins on the MMA-critical
  barriers): WASH (4.48 vs 4.51 at mma_n=32) — try_wait wake latency was
  never the bottleneck. Knob kept, default off.
- Stream phase FLAT vs mma_n → activation re-reads (80 CTAs × same slice) hit
  L2 for free. **4-CTA cluster activation multicast is DEAD** (no bytes to
  save that aren't already nearly free).
- The epilogue is the scaling bottleneck: ~31 ns per serial token per thread.
  Fix: more epilogue warpgroups (`epi_split`, warps w+4k share the same 32
  tmem lanes on disjoint token slices). Sweep (probe_v7_episplit.py, fused,
  bit-identical across splits):

| M | mma_n | old (EPI1/2) | EPI4 | EPI8 | v5 fused |
|---|---|---|---|---|---|
| 32 | 32 | 6.27 (E1) | **5.44** | — | 6.21 |
| 48 | 64 | 6.69 (E2) | **6.14** | 6.24 | 7.07 |
| 64 | 64 | 6.75 (E2) | **6.18** | 6.34 | 7.30 |
| 96 | 96 | 7.71 (E2) | 6.85 | **6.78** | 6.56 |
| 128 | 128 | 8.80 (E2) | 7.62 | **7.49** | 7.71 |

  New defaults: EPI4 for mma_n=32/64, EPI8 for 96/128. v7 now beats v5
  standalone at 32/48/64/128 and is within noise at 96.

## Next / open

1. ~~v7 chain numbers~~ — **done** (see chain table): after the drain-path
   restructure (xn preloaded to registers before the MMA-drain wait; ALL warps
   wait on the commit barrier instead of warp-1-waits-then-bar.sync), v7 wins
   M=2–64 outright in-chain. Chain dispatch: v3 M=1, v7 2–64, v5 96–192.
2. ~~M=96/128 v7 GEMM gap~~ — **closed by epi_split** (see phase decomposition:
   the gap was the serial epilogue, not the stream; EPI4/EPI8 fixed it). 4-CTA
   cluster activation multicast ruled out (activation re-reads already hit L2
   for free); mma_n=64 × grid.y=2 ruled out (2-wave cliff).
3. **Upstream PR material (down kernel)**: early `launch_dependents` (1.4–3.0 µs fused
   chain gain, 0.3–0.5 µs unfused) + M=4 register-residency fix (`min_blocks_per_mp=3`,
   -20% at M=4). Both validated, bit-identical; touches the merged kernel.

## Phase 4: upstream-link PDL (3-kernel chain, bench_hc_chain3.py)

Production mHC block: `hc_combine_norm -> hc_down_silu -> up -> hc_gate_mix`.
`hc_combine_norm` (Triton) already runs with PDL but fires
`gdc_launch_dependents` ~60% in (after the combine stores, before the norm
tail). Prototype `hc_combine_norm_early` (workspace copy, bit-identical
asserted) fires at CTA start; down (early-PDL variant) then launches at
~1.06 µs instead of ~2.1–2.6 and pre-streams weights under combine_norm.

Totals (µs), `results_hc_chain3.json`:

| M | cnprod | cnearly | cnearly+down-prefetch |
|---|--------|---------|-----------------------|
| 1  | 9.47  | 9.54  | 9.47  |
| 2  | 10.18 | 10.02 | 9.70  |
| 4  | 10.59 | 10.66 | 10.66 |
| 8  | 11.15 | 11.26 | —     |
| 16 | 11.71 | 11.71 | —     |
| 32 | 12.13 | 11.90 | —     |
| 48 | 13.44 | 13.26 | —     |

**Verdict: wash (all deltas within the ±0.3–0.5 µs noise band).** Down
launches ~1.2 µs earlier but stretches by the same amount; the chain total
is conserved. This retroactively explains the merged down-side work's
PDL ≈ 0 on the cn→down link. Same law as Phase 3a: at these sizes every
kernel sits at its own latency/stream floor, so overlap only redistributes
time between kernels. The one structural idea left — a single fused
down+up kernel (one launch, one fixed cost) — has a ceiling of ~1 µs at
M=1 and would face the same capped-grid bandwidth problem as Phase 3a for
its down phase; assessed as not worth the build.

## Resolved findings (dead ends and validated conclusions — don't re-try)

- **v7 drain-path restructure** — validated, kept: xn gate values preloaded to
  registers BEFORE the MMA-drain wait (hides the smem reads under the drain),
  and all warps wait on `bar_mma_done` directly (one less wake+bar.sync hop).
  −0.2–0.5 µs at M≤16 fused; bit-identical (md_v6=0.0000).
- **v7 GEMM-base decomposition** (see phase table): the base is at the
  hardware floor — stream 3.78 µs (HBM) + tcgen05 drain/commit ~0.5 µs
  (`late_mma_1`: a single post-stream MMA already costs +0.5) + epilogue.
  `no_epi` ≈ nvjet's full GEMM at every M. No further base headroom except
  PDL overlap (done).
- **gemm_only epilogue** — scalar strided gmem stores replaced with staged
  TMA store (4 stream-quarter boxes); md=0.0000 vs nvjet. Bank-conflicted
  across stream quarters, unfixable by padding (TMA box bases need 128B
  alignment == bank period). Scaffold only; the fused path's (32, MMA_N)
  staging is conflict-free (only stream-0 lanes store).
- **EPI splits in gemm_only** — tried, WORSE (EPI warpgroups share the same
  32 tmem lanes so tmem reads don't parallelize; 128×EPI threads inflate
  launch/init). gemm_only and non-"full" probes force EPI1.
- **spin_wait (mbarrier.test_wait spins)** — WASH; try_wait wake latency is
  not the bottleneck. Knob kept, default off.
- **Early tmem dealloc (right after the last tcgen05.ld, overlapping the
  mix+staging+TMA-store tail)** — WASH standalone (4.80/4.86/5.06/5.41 vs
  4.77/4.83/5.06/5.44 at M=1/8/16/32, all within noise; M=96 slightly worse
  from holding all rG chunks live). dealloc is cheap and CTA exit wasn't
  gated on it. Reverted. And the drain itself is NOT hideable this way:
  the ~0.5 µs (late_mma_1) sits between the last weight byte and the first
  acc read — only acc-independent work can fill it, and xn preload already
  does. (Split-acc K-halves would let half the acc load during the drain
  but breaks bit-identity via K reassociation — rejected.)
- **no_fence** — WASH: per-stage `fence_after_thread_sync` is free; kept for
  idiom consistency.
- **Early `launch_dependents` in down kernel** — validated (see chain table and open
  item 3).
- **Down-kernel M=4 anomaly** — root-caused: ptxas allocates
  220 regs/thread → 2 CTAs/SM → 304 slots < 324-CTA grid → ragged 2nd wave. Fixed via
  `min_blocks_per_mp=3`. Side finding: production's `--ptxas-options -maxrregcount=64`
  is **silently ignored** by the DSL (cubin byte-identical with/without).
- **v5 TMA EVICT_FIRST weight hint** — no win: neutral at
  small M, up to 4.3% worse from M≈96 (kills cross-token-tile L2 weight reuse).
- **v7 init sequence (gn-kernels pattern)** — TMA descriptor prefetch moved to
  warp 1 (X/O on warp 2, fused only), concurrent with warp 0's mbarrier init
  instead of serialized after the barrier; `mbarrier_init_fence` dropped:
  `cp.async.bulk` accesses its mbarrier operand via the *generic* proxy (PTX
  ISA `cp.async.bulk.tensor`), so with 1-SM/CTA-local barriers the `bar.sync`
  after init already publishes them — the fence is a cluster-scope release and
  only matters for cta_group=2 / cluster kernels (which is why gn-kernels and
  the in-repo DSL pipelines keep it unconditionally). Bit-identical; chain and
  standalone numbers unchanged within noise.
- **v7 TMA issue order** — auto-selected by `use_pdl` (PDL → all weights
  pre-wait, activations post-wait; non-PDL → per-stage W/A interleave, worth
  ≤0.22 µs standalone). The `interleave_wa` knob is removed; `pdl_wait_first`
  is kept as a probe. Chain attribution (table above): early-launch overlap
  ~0.8–1.2 µs, weight pre-stream ~0.2–0.6 µs, additive.
- **Phase 3a: capped-down co-scheduling** — DEAD (3 variants, all lose to the
  uncapped chain at M=1/2/4; `probe_down_cap_chain.py`,
  `results_down_cap_chain.json` / `results_down_cpasync_chain.json`).
  Mechanism validated: PDL-launched up CTAs are only placed on SMs with *no*
  resident primary CTAs — with ≤108 down CTAs the up kernel starts at
  ~1.5–1.9 µs (vs ~3.0 uncapped). But down is load-latency-concurrency-bound:
  every capped variant stretches down more than the early up start gains.
  Variants tried, down dur at M=1 (uncapped 3.84): dynamic strided loop
  cap108 = 6.08; static-unrolled two-phase (barrier-free dot region) J=3 =
  5.86 (ptxas still won't keep cross-column LDGs in flight; registers bound
  in-flight bytes); cp.async smem pipeline (S=4 stages, per-thread commit
  groups, bit-identical) J=3 = 5.70. Chain totals: best capped = 8.32 vs
  7.62 uncapped at M=1. Conclusion: the chain is bound by *total* stream
  bytes through one memory system; moving bytes between kernels is ~1:1.
  Files kept: `_hc_down_silu_fma_capped.py` (LDG two-phase),
  `_hc_down_silu_fma_cpasync.py` (cp.async), wrapper knobs
  `fma_cols_per_cta` / `fma_cpasync` in `hc_down_silu_early.py`.
- **v7 `xn_pre_wait`** — wash (`probe_v7_xn_pre.py`, `results_v7_xn_pre.json`;
  M=16/32/48: 9.41/10.02/11.07 vs 9.38/9.90/11.04 baseline). The post-wait xn
  TMA is already hidden behind the MMA window. Knob kept, default off. NOTE:
  pre-wait xn is only race-free in production if the down kernel fires
  `launch_dependents` after its own `griddepcontrol_wait` (otherwise up CTAs
  can be live while xn's producer — down's predecessor — still runs).
- **v8 register-accumulator mma.sync** — REJECTED on roofline math, no build:
  per-CTA MMA work at M=32 is 128 rows × 32 tok × 320 K × 2 ≈ 2.6 MFLOP;
  legacy mma.sync bf16 (~2048 FLOP/cyc/SM) needs ~1270 cyc ≈ 0.73 µs — the
  same as tcgen05's measured 0.74 µs MMA+drain phase, and strictly worse
  beyond M=32. The tmem alloc/drain saving (~0.3–0.5 µs) is eaten by the
  slower MMA path at every M where tcgen05 is dispatched (8–64).
- **v7 EVICT_FIRST weight hint** — also neutral (`weight_evict_first` knob,
  `/tmp/probe_v7_evict.py`): identical at every M=1..96 (5.02/5.02, 6.85/6.85,
  7.89/7.87). Unlike v5 there is no weight reuse to kill, but the whole working
  set (~9 MB) fits L2 anyway, so the hint has nothing to do. Knob left in place,
  default off.
- **v7 3D weight TMA (stream-innermost fill)** — bit-identical and
  perf-identical at every M (`/tmp/probe_v7_w3d.py`, gemm-only; e.g. M=64:
  6.78 vs 6.75, M=128: 8.93 vs 8.96). The weight stream is bound by per-CTA
  bytes, not by TMA box shape/dim count — so the 4D interleave is free, and
  simplifying it buys nothing. **Knob removed after the neutral result**; the
  4D path is the only one in the tree.
- **v5 epilogue xn TMA-staging** — no win for v5: per-thread LDG
  prefetch lands xn in registers during the MMA window; SMEM routes lose at every M.
  (v7 differs: its xn TMA overlaps the mainloop and wins — the difference is v7's
  LDG prefetch serialized at the 255-reg cap.)
- **v5 M≥224 wave cliff** — partially solved: mn96 for
  224–256, multi-pass `m_passes=2` for 257–384 → tie with baseline; 448/512 structural
  (tmem-capacity ≥2 waves). Failed: mma_n=112, 2-CTA/SM (register-file blocked),
  stream-rate variants (~5.9 µs/CTA cold stream is a machine floor). Variant forks
  (`_up_gate_mix_v5_lm.py` etc.) pruned in the stash cleanup; conclusions live here
  and in `RESULTS_FUSED_UP.md`.
- **Triton `hc_gate_mix` already optimal** (`bench_gate_mix.py`): prod config
  best-or-tied at every M; the ~3.4 TB/s at M=512 is the triton plateau.
- **CuteDSL gate-mix** — parity only, not adopted (`gate_mix_cute.py`). Traps: never
  use bare `cute.math.exp` in an epilogue (full-precision expf path) — use
  tanh.approx (`0.5*tanh(x/2, approx=True)+0.5`, one SFU op) or `exp2(approx, ftz)`.
  v3/v5/v7 all default to `sigmoid_mode="tanh"`.
- **v5 base GEMM vs nvjet** (`probe_v5_gemm_only.py`): v5_gemm bit-identical but loses
  at every M (M=1: +0.9, M=128: +3.1); dm2 pipe floor ~5.4 µs at small M. The fused
  epilogue is cheaper than a plain GEMM store (4× fewer bytes). Fusion win = removing
  the gate-mix kernel, not a better GEMM.
- **v5 design review**: MMA_M=128 strictly worse for v5 (halves channel grid + ring
  depth); split TMA warps earn their keep (activations land 0.2–0.4 µs earlier);
  TMA descriptor prefetch not available in the experimental API (atom API only).
- **nvjet SASS** (ncu JIT capture): CTA tile 128ch×8tok, BK=64, 12-stage ring,
  4-CTA cluster with activation TMA multicast, grid=80, 256 threads (w4 = single TMA
  warp for both operands, w6 = MMA), 20 UTCHMMA on ONE 128×8 accumulator,
  stmatrix+UTMASTG epilogue, PREEXIT right after last TMA. Motivated v7's design.
- **v7 GEMM-only** (µs, cold-L2, PDL off; bit-identical to cuBLAS all M):

  | M | nvjet | v5 | v6 | v7 |
  |---|---|---|---|---|
  | 1 | 4.48 | 5.41 | 5.54 | **4.70** (at the TMA floor) |
  | 8 | 4.51 | 5.54 | 5.76 | 4.86 |
  | 32 | 4.90 | 6.43 | 6.56 | 5.66 |
  | 64 | 4.77 | 8.10 | 7.30 | 6.78 |
  | 96 | 5.28 | 6.53 | 6.66 | 7.84 |
  | 128 | 5.02 | 8.13 | 7.23 | 8.99 |

- **v7 fused epilogue design notes**: scalar-LDG xn prefetch serializes at the
  255-reg cap (chunked double-buffering made it worse); xn TMA→smem fixed it.
  tcgen05.ld vector length must be pow2 dividing the token tile (lowbit, cap 64).
  Epilogue token loop is serial per thread → MMA_N≥64 uses 2 warpgroups (warps w
  and w+4 share 32 tmem lanes, split tokens); this made the epilogue ~free.

## Files

- `fused_up/_up_gate_mix_v3.py` — FMA (small-M candidate)
- `fused_up/_up_gate_mix_v4.py` — mma.sync (dropped, kept for reference)
- `fused_up/_up_gate_mix_v5.py` — tcgen05 (large-M candidate)
- `fused_up/_up_gate_mix_v6.py` — tcgen05, raw standard-API (no tiled_mma /
  make_tiled_tma_atom_A/B): generic make_tiled_tma_atom + simple_tma_copy +
  raw `_tcgen05` idesc/sdesc/mma_f16/ld, interleaved tmem packing + shuffle mix,
  `debug_raw`/`gemm_only` knobs; wrapper `UpGateMixV6`
- `fused_up/_up_gemm_v7.py` — v7: MMA_M=128 (4 streams x 32 ch in one acc),
  80 CTAs, 5-stage unrolled no-wrap ring, 4 warps (8 in the fused epilogue
  for MMA_N≥64), 4D weight TMA (8-row stream-interleave); fused epilogue:
  xn TMA→smem, butterfly mix, TMA store; `gemm_only` scaffold mode;
  wrapper `UpGemmV7` in the same file
- `fused_up/up_gate_mix.py` — wrappers (v3–v6 dispatch); `_up_gate_mix.py` /
  `_up_gate_mix_v2.py` kept only for its legacy lazy imports
- `fused_down/early_pdl/` — down-kernel copies with `launch_dependents` at CTA start
  (`hc_down_silu_early.py` dispatcher, bit-identical output); also the Phase-3a
  probes `_hc_down_silu_fma_capped.py` (LDG two-phase capped grid),
  `_hc_down_silu_fma_cpasync.py` (cp.async staged capped grid), and the Phase-4
  `hc_combine_norm_early.py` (Triton, early `gdc_launch_dependents`)
- Benches/probes: `bench_fused_up_v{3,4,5}.py`, `bench_fused_up_v5_largem.py`,
  `bench_v5_hc_chain.py` (also hosts `measure_graph`/`summarize`),
  `bench_v3_hc_chain.py`, `bench_hc_chain_pdl.py` (PDL round + v7 ordering
  probes), `bench_hc_chain_ds.py` (3-kernel down→up→ds chain, downstream PDL
  edge), `bench_hc_chain3.py` (cn→down→up chain, Phase 4), `bench_up_floor.py`
  (baseline split), `bench_gate_mix.py` (triton gate-mix sweep),
  `probe_stream_floor.py`, `probe_v5_gemm_only.py`, `probe_v6{,_gemm_only}.py`,
  `probe_v7_smoke{,_pdl}.py`, `probe_v7_gemm_only.py`, `probe_v7_fused.py`,
  `probe_v7_phases.py`, `probe_v7_episplit.py`, `probe_v7_xn_pre.py`,
  `probe_down_cap_chain.py`, `probe_gate_mix_cute.py`, `gate_mix_cute.py`
- Data: `results_fused_up_v{3,4,5}.json`, `results_fused_up_v5_largem.json`,
  `results_v5_hc_chain.json`, `results_v3_hc_chain.json`, `results_hc_chain_pdl.json`,
  `results_hc_chain_ds.json`, `results_hc_chain3.json`, `results_up_floor.json`,
  `results_stream_floor.json`, `results_down_cap_chain.json`,
  `results_down_cpasync_chain.json`
- e2e methodology: `vigil_hc_down_silu_ab_{8k1k,latest}.yaml`,
  `vigil_hc_down_silu_acc_ab.yaml` (down-fusion A/B; numbers quoted in the e2e
  section). The 1.1 GB `runs/` vigil logs and SASS `dump/` were NOT kept.
- Full v1–v5 history: `RESULTS_FUSED_UP.md`
- Pruned in the stash cleanup (dead ends, conclusions kept above): v1/v2-era
  benches, all v4/v5 sweep + ablate scripts, `_v5_{ef,lm,xn}` / `_v3_dbg`
  variant forks, one-off `check_*`/`probe_v5_*` scripts, and their stale
  results/logs.
