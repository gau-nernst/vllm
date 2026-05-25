# KDA Optional Backend Setup

These steps make the local `cuLA/` and `FlashKDA/` clones usable from `kda_triton.py` for the prefill speed benchmark.

## Environment

Run everything from the vLLM checkout root:

```bash
cd /home/thien/vllm/kda_dev
```

Use the project venv and `uv`; do not use system `python3` or bare `pip`.

## cuLA

Initialize cuLA submodules. The Blackwell path imports FLA from cuLA's local submodule, and the Hopper path also needs cuLA's CUTLASS submodule when building `cula.cudac`.

```bash
git -C cuLA submodule update --init csrc/cutlass third_party/flash-linear-attention
```

cuLA currently pins CUTLASS DSL 4.4.2. The venv had 4.5.0, which fails the Blackwell CuTeDSL compile with `cutlass.cute.arch.ProxyKind` missing. Pin it before running the cuLA backend:

```bash
uv pip install nvidia-cutlass-dsl==4.4.2
```

`kda_triton.py` benchmarks cuLA's non-fully-fused forward path, `cula.kda.chunk_fwd.chunk_kda_fwd`. The public `cula.kda.chunk_kda` wrapper imports the backward extension too; for this forward-only speed table we call the forward function directly. The Blackwell fully fused prefill path is still WIP and prints CuTe/CUTLASS layout diagnostics during compile, so it is intentionally not used.

## FlashKDA

FlashKDA's original `setup.py` only built `sm_90a`. For GB200, add an `sm_100a` gencode path, then rebuild in the current venv:

```bash
FLASH_KDA_DISABLE_SM90=1 NVCC_THREADS=8 \
  uv pip install --force-reinstall --no-deps --no-build-isolation -e FlashKDA
```

`--no-build-isolation` is required because `FlashKDA/setup.py` imports `torch` during build setup.

## Verification

Backend import smoke test:

```bash
.venv/bin/python - <<'PY'
import kda_triton
print("cuLA", kda_triton.cula_chunk_kda_fwd is not None)
print("FlashKDA", kda_triton.flash_kda is not None)
PY
```

Expected on this GB200 setup:

```text
cuLA True
FlashKDA True
```

Run the speed benchmark:

```bash
.venv/bin/python kda_triton.py
```

Notes:

- `kda_triton.py` does not modify the Triton prefill kernel for these backends.
- The benchmark precomputes gate and beta for FLA, Triton dev, and cuLA before timing. FlashKDA still computes gate and beta inside its prepare kernel because the public API takes raw logits.
- FlashKDA executed successfully in the smoke test after the `sm_100a` rebuild. Its prepare kernel also performs q/k L2 normalization; there is no public disable flag in the current API.
- The benchmark uses cuLA's non-fully-fused forward path. The direct `chunk_kda_fwd` path used here receives already-normalized q/k and precomputed gate/beta; cuLA's public wrapper can apply q/k L2 normalization with `use_qk_l2norm_in_kernel=True`.

## Backend Subkernels

This is the current mental model for the local backend implementations used by `kda_triton.py`.

### FlashKDA

FlashKDA launches two CUDA kernels from `FlashKDA/csrc/smxx/fwd_launch.cu`:

1. `_flash_kda_fwd_prepare`
2. `_flash_kda_fwd_recurrence`

`_flash_kda_fwd_prepare` is launched with grid `(total_tiles, H)`, so each CTA handles one `(sequence chunk, head)` tile. It performs the per-chunk preprocessing and writes a workspace for the recurrence kernel.

Main work in the prepare kernel:

- TMA-load `q`, `k`, raw `g`, beta logits, and `dt_bias`.
- L2-normalize `q` and `k` in-kernel.
- Compute safe gate activation and chunk-local cumsum:
  `lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))`, stored in base-2 scale.
- Compute decayed/restored forms used by KDA:
  `k_decayed`, `q_decayed`, `k_restored`, and `g_total`.
- Compute local KKT and QK terms with warp-level `mma.sync` helpers.
- Apply beta sigmoid to KKT, build the triangular inverse, and store workspace tensors.

Prepare-kernel workspace tensors:

- `ws_kd`: decayed K, shape `[H * total_tiles, CHUNK, D]`.
- `ws_qd`: decayed Q, shape `[H * total_tiles, CHUNK, D]`.
- `ws_kr`: restored K, shape `[H * total_tiles, CHUNK, D]`.
- `ws_gt`: total chunk gate decay, shape `[H * total_tiles, D]`.
- `ws_inv`: triangular inverse, shape `[H * total_tiles, CHUNK, CHUNK]`.
- `ws_mqk`: masked/scaled QK tile, shape `[H * total_tiles, CHUNK, CHUNK]`.

`_flash_kda_fwd_recurrence` is launched with grid `(N, H)`, so each CTA handles one full sequence and head. It streams over that sequence's chunks, loads the prepare-kernel workspace plus `v`, and computes the recurrent state/output.

Main work in the recurrence kernel:

- TMA-load `v`, beta, the six workspace tensors, and optional initial state.
- Use `ws_inv`, `ws_kd`, `ws_kr`, and `v` to form the KDA update terms.
- Advance the recurrent state across chunks using `ws_gt`.
- Combine the state contribution and local `ws_mqk` contribution to produce `out`.
- Store final state when requested.

Relevant files:

- `FlashKDA/csrc/smxx/fwd_launch.cu`
- `FlashKDA/csrc/smxx/fwd_kernel1.cuh`
- `FlashKDA/csrc/smxx/fwd_kernel2.cuh`
- `FlashKDA/csrc/smxx/utils.cuh`

### CuLA Non-Fully-Fused Path

The benchmark uses CuLA's non-fully-fused forward path, `cula.kda.chunk_fwd.chunk_kda_fwd`. This is Python orchestration over several compiled kernels.

The high-level stage sequence is:

1. Gate cumsum setup.
2. KDA intra chunk kernel.
3. Recompute `w`/`u`/`kg`.
4. Chunk state recurrence.
5. Output kernel.

Stage details:

- Gate setup in `cuLA/cula/kda/chunk_fwd.py`:
    - If `use_gate_in_kernel=True`, calls FLA's `kda_gate_chunk_cumsum(...)` to compute safe gate plus chunk cumsum.
    - In the benchmark we pass precomputed safe gate, so CuLA calls `chunk_local_cumsum(...)` with `RCP_LN2` scaling.

- Intra chunk in `cuLA/cula/kda/chunk_intra.py`:
    - Calls `cula_cuda.chunk_kda_fwd_intra_cuda(...)`.
    - Produces `Aqk` and `Akk` for each 64-token chunk.
    - The SM100 implementation decomposes a 64-token chunk into four 16-token subchunks.
    - For KKT/QK, it prepares pair-specific scaled operands and issues grouped `tcgen05` MMAs:
        - 3 off-diagonal row-group calls per K-slice: `N16`, `N32`, `N48`.
        - 4 diagonal calls per K-slice: one `N16` call per diagonal subchunk.
        - With `HeadDim=128` and `TileK=32`, this is `7 * 4 = 28` `tcgen05.mma` calls per chunk/head for the intra QK/KKT stage.

- Recompute `w`/`u`/`kg` in `cuLA/cula/kda/chunk_intra.py`:
    - Calls `cula_cuda.recompute_w_u_cuda(...)` after `Akk` is available.
    - Produces the transformed update tensors consumed by the recurrence stage.

- State recurrence in `cuLA/cula/ops/chunk_delta_h.py`:
    - Called as `chunk_gated_delta_rule_fwd_h(...)` from `chunk_fwd.py`.
    - Advances the recurrent KDA state across chunks and returns `h`, `v_new`, and final state.

- Output in `cuLA/cula/ops/fwd_o.py`:
    - Called as `chunk_gla_fwd_o(...)` from `chunk_fwd.py`.
    - Computes the final output from `q`, `v_new`, cumulative gate `g`, local `Aqk`, and recurrent state `h`.

Relevant files:

- `cuLA/cula/kda/chunk_fwd.py`
- `cuLA/cula/kda/chunk_intra.py`
- `cuLA/csrc/kda/sm100/kda_fwd_intra_mainloop_sm100.hpp`
- `cuLA/csrc/kda/sm100/fwd_helpers.hpp`
- `cuLA/cula/ops/chunk_delta_h.py`
- `cuLA/cula/ops/fwd_o.py`

The practical difference is that FlashKDA exposes a two-kernel end-to-end forward, while CuLA's non-fully-fused path exposes a sequence of specialized kernels. CuLA's SM100 intra stage is much more complicated because it uses Blackwell `tcgen05` with extra operand preparation for stable pair-dependent anchors. FlashKDA's prepare kernel is easier to reason about: it uses `mma.sync`-style local tiles and stores explicit workspace for a separate recurrence kernel.
