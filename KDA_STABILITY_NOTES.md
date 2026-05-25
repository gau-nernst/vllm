# KDA Stability Notes

This note captures the numerical-stability trick used by cuLA and FlashKDA that the baby Triton prefill in `kda_triton.py` does not currently implement internally.

## The Risk

The KDA recurrence repeatedly forms exponentials of cumulative gates inside each chunk:

```text
g_cu = cumsum(g, dim=time)
ka = k * exp(g_cu)
kb = k * exp(-g_cu)
h_scaled = h * exp(g_last)
kg = k * exp(g_last - g_cu)
qa = q * exp(g_cu)
```

If `g` is produced by the vanilla KDA gate

```text
g = -exp(A_log) * softplus(raw_g + dt_bias)
```

then `g <= 0`, but the magnitude is not explicitly bounded. Over a chunk, `g_cu` can become very negative, so `exp(-g_cu)` can become very large. That is the unstable direction for the `kb` term and the `qk/kkt` products.

## The Safe-Gate Variant

cuLA and FlashKDA use a lower-bound sigmoid gate for their safe/fused paths:

```text
g = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))
```

with the recommended/default

```text
lower_bound = -5.0
```

Because `sigmoid(x) in (0, 1)`, this guarantees

```text
g in (lower_bound, 0) = (-5, 0)
```

So for a chunk of length `C`, the cumulative gate is bounded:

```text
g_cu in (C * lower_bound, 0)
```

For `C = 16`, that means roughly `g_cu in (-80, 0)`. For `C = 64`, roughly `(-320, 0)`. This is still a large range, but it is predictable and prevents one bad raw gate from making the exponentials arbitrarily large.

## cuLA

cuLA exposes this through `use_gate_in_kernel=True`, `safe_gate=True`, and `lower_bound=-5.0`.

In its Python/CuTe paths, cuLA calls FLA's `kda_gate_chunk_cumsum(...)` when gate fusion is enabled. That function does two things together:

```text
gate = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))
g_cu = cumsum(gate, dim=time within chunk)
```

It also multiplies by `RCP_LN2`, so the downstream kernels can use base-2 exponential machinery consistently with FLA/Triton helpers:

```text
g_cu_log2 = cumsum(gate) * log2(e)
```

Relevant local files:

- `cuLA/cula/kda/hopper_fused_fwd.py`
- `cuLA/cula/kda/blackwell_fused_fwd.py`
- `cuLA/third_party/flash-linear-attention/fla/ops/kda/gate.py`

## Sub-Chunk Recentering In FLA And cuLA

The lower-bound gate alone does not make `chunk_size=64` safe if the kernel still computes the unstable direction over the whole chunk. With `lower_bound=-5`, a 64-token chunk can have:

```text
cumsum(gate) ~= -320
exp(-cumsum(gate)) ~= exp(320)
```

That overflows fp32. The missing trick is that FLA and cuLA do not compute the sensitive intra-chunk exponentials with one 64-token reference. They split the 64-token chunk into four 16-token subchunks and recenter the gate inside each subchunk.

For the diagonal 16x16 subchunk, FLA loads the cumulative gate `g` for that subchunk, chooses a local anchor near the middle of the subchunk, and computes:

```text
g_mid = g[subchunk_start + min(BC / 2, remaining_tokens - 1)]
g_rel = g - g_mid

q_scaled = q * exp2(g_rel)
k_scaled = k * exp2(-g_rel)
```

with `BC = 16`. This bounds the largest exponent by the distance from the middle of a 16-token block, not by the full 64-token chunk. For the safe gate lower bound `-5`, the worst natural-exp span is roughly `8 * 5 = 40`, which is large but below fp32 overflow. The local FLA file even comments that this subtraction keeps the exponent below the `exp2` overflow range.

Relevant FLA code:

- `cuLA/third_party/flash-linear-attention/fla/ops/kda/chunk_intra.py`
- `chunk_kda_fwd_intra(...)` sets `BT = chunk_size` and `BC = 16`.
- `chunk_kda_fwd_kernel_intra_sub_chunk(...)` computes `b_gm = b_g - b_gn`, then `exp2(b_gm)` and `exp2(-b_gm)`.

For off-diagonal blocks, the chunk is treated as a 4x4 lower-triangular matrix of 16-token subchunks:

```text
          j=0        j=1        j=2        j=3
i=0      intra
i=1      inter      intra
i=2      inter      inter      intra
i=3      inter      inter      inter      intra
```

The off-diagonal terms are also expressed relative to subchunk anchors. Suppose the query tile comes from a later subchunk `S2` and the key tile comes from an earlier subchunk `S0`:

```text
S0 = tokens  0..15
S1 = tokens 16..31
S2 = tokens 32..47
S3 = tokens 48..63
```

Let:

```text
g_j = cumulative gate for a key token in S0
a   = anchor, usually the first cumulative gate of S2
g_i = cumulative gate for a query token in S2
```

Because each safe-gate increment is non-positive, the cumulative gate is monotone non-increasing over time. Therefore, for this causal off-diagonal tile:

```text
g_j >= a >= g_i
```

The desired decay factor is:

```text
exp2(g_i - g_j)
```

FLA/cuLA split it through the anchor:

```text
exp2(g_i - g_j) = exp2(g_i - a) * exp2(a - g_j)
```

and apply the two factors on opposite MMA inputs:

```text
q_scaled = q_i * exp2(g_i - a)
k_scaled = k_j * exp2(a - g_j)
```

Given the ordering `g_j >= a >= g_i`, both exponents are non-positive:

```text
g_i - a <= 0
a - g_j <= 0
```

So both factors are at most `1`. The query-side factor is also local to the destination subchunk, so its magnitude is bounded by roughly one 16-token span. The key-side factor may span several subchunks, but it is in the decay direction and can only underflow toward zero. That is safe; the dangerous case is a large positive exponent.

This is the important off-diagonal distinction from the baby Triton implementation. A naive full-chunk formulation may materialize `exp2(-g_j)` or `exp2(-g_cumsum)` for late positions, which can become a huge positive exponent. The anchored off-diagonal formulation keeps the algebraic product the same while making each individual exponential a decay.

CuLA follows the same structure. Its Python/Triton path has the same `BT=64`, `BC=16` split and the same `b_g - b_gn` diagonal recentering. Its Blackwell CuTe path documents the 4x16 layout explicitly:

```text
if i > j:  B = exp2(g_first_i - g_j[x]) * K_j[x]
if i == j: B = exp2(g_half_i  - g_i[x]) * K_i[x]
```

So for off-diagonal blocks, CuLA uses the first gate of the destination subchunk as the reference. For diagonal blocks, it uses the midpoint gate of that same 16-token subchunk. This keeps the growth direction local while the full 64-token chunk is assembled from relative block factors.

Relevant cuLA code:

- `cuLA/cula/kda/chunk_intra.py`
- `cuLA/csrc/kda/sm100/kda_fwd_intra_mainloop_sm100.hpp`

The state/output phases are less dangerous because they mostly use the decay direction:

```text
h *= exp2(g_last)
qg = q * exp2(g)
```

With safe gate, `g <= 0`, so these terms underflow toward zero at worst. The dangerous term is the inverse decay, such as `exp2(-g)`, and that is where the subchunk recentering matters.

## FlashKDA

FlashKDA expects raw gate logits and beta logits. It fuses both preprocessing steps into the CUDA kernels:

```text
gate = lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias))
g_cu = cumsum(gate, dim=time within chunk)
beta = sigmoid(beta_logits)
```

The C++ binding computes:

```text
gate_scale = lower_bound * log2(e)
```

and the kernel computes the gate/cumsum in one pass. Its sigmoid implementation uses a tanh approximation:

```text
sigmoid(x) ~= 0.5 * tanh.approx(0.5 * x) + 0.5
```

Relevant local files:

- `FlashKDA/flash_kda/__init__.py`
- `FlashKDA/csrc/flash_kda.cpp`
- `FlashKDA/csrc/smxx/fwd_kernel1.cuh`
- `FlashKDA/csrc/smxx/fwd_kernel2.cuh`
- `FlashKDA/csrc/smxx/utils.cuh`

## What The Baby Triton Kernel Does

The baby Triton kernel in `kda_triton.py` currently assumes its `g` input is already activated. It then directly does:

```text
g_cu = cumsum(g)
ka = k * exp(g_cu)
kb = k * exp(-g_cu)
...
```

So it does not enforce the lower-bound gate itself. If callers pass the vanilla softplus gate, the kernel inherits that wider dynamic range. In the benchmark harness we currently precompute the lower-bound gate outside the baby Triton kernel for fair comparison, but the kernel itself does not protect against unsafe `g` inputs.

## If We Want The Triton Kernel To Match

The baby Triton prefill should take raw `g`, `A_log`, and `dt_bias`, then compute the safe gate before the local cumulative sum:

```text
gate = -5.0 * sigmoid(exp(A_log) * (raw_g + dt_bias))
g_cu = cumsum(gate)
```

Optionally, it can also take beta logits and compute:

```text
beta = sigmoid(beta_logits)
```

That would align the semantics with FlashKDA and cuLA's `safe_gate=True` path and avoid relying on callers to supply already-safe activated gates.
