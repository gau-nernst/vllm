# [Perf][PP] Send copies of PP outputs instead of waiting on the previous send

<!-- DRAFT for review. Placeholders marked TODO(results) are filled from the
MiniMax-M3-MXFP8 A/B run (vigil m3_mxfp8_pp2_send_copy_ab, 2026-10-27). -->

## Purpose

With pipeline parallelism, each non-last stage currently waits at the start of
every step for its previous step's intermediate-tensor send to finish (#54962).
That wait exists because the next forward can overwrite the tensors the send
is still reading. CUDA graph outputs are static buffers (FULL graphs return a
slice of a persistent buffer, PIECEWISE graphs return memory from the graph
pool), and MRV2 relays aux hidden states as views of the persistent receive
buffer. #49274 had removed the wait and produced garbage outputs.

The wait is expensive. An NCCL send completes only once the next stage posts
the matching receive, and ProcessGroupNCCL orders that receive after the next
stage's in-flight compute. So stage *k* idles until stage *k+1* finishes its
previous microbatch. In prefill with uneven step costs, the stages take turns
blocking each other.

This PR sends a copy of the outputs instead and drops the wait:

- `execute_model` sends `{k: v.clone() for k, v in output.tensors.items()}`.
  The copy runs on the compute stream. ProcessGroupNCCL orders its send after
  the compute stream, and `GroupCoordinator.isend_tensor_dict` already keeps
  sent tensors alive until their sends complete (`_pending_isends`). So the next
  forward can overwrite the outputs at once, and nothing waits.
- The copies outlive the forward pass, so the profile run does not see them.
  The worker now reserves `(max_concurrent_batches + 1) x` the bytes the profile
  run's largest batch sends, on non-last PP ranks. This covers one copy per
  batch in flight, plus the next copy, which is made before finished sends are
  released. The size comes from the profile run's actual payload (V1 and V2
  runners), not `make_empty_intermediate_tensors`. That function sizes what a
  rank *receives*, which can be smaller than what it sends (DeepStack
  payloads, relayed aux hidden states).

The copy costs one device-to-device copy per send. For MiniMax-M3 that is
TODO(results) per 16K-token batch, against a stage pass of TODO(results).

### KV cache sizing

The reserve changes how the KV cache is **sized** at startup, not how much
memory is used while serving. On non-last PP ranks, the available KV cache
memory drops by the reserve. For MiniMax-M3-MXFP8 PP2 with
`--max-num-batched-tokens 16384`, that is TODO(results) GiB on stage 0
(TODO(results) -> TODO(results) GiB available KV cache memory). Deployments
that tuned `--gpu-memory-utilization` or `--kv-cache-memory-bytes` tightly
may need to adjust them. With `--kv-cache-memory-bytes` set, nothing is
reserved automatically, as before.

## Not duplicating existing work

- #54962 added the wait this PR replaces.
- #63745 (draft) adds a bounded snapshot ring as part of an opt-in dynamic-PP
  runtime and keeps the default send path. This PR changes the default path
  and needs no new runtime or flag.
- #55058 is an opt-in ROCm FP8 PP transport (`VLLM_ROCM_PP_TRANSPORT`).
- #53948 defers sampled-result receives, which is a different hand-off.

## Test Plan

- `pytest tests/v1/worker/test_gpu_worker.py`.
  `test_execute_model_waits_previous_pp_send_before_forward` asserted the
  removed wait. It is replaced by `test_execute_model_sends_copies_of_pp_outputs`,
  which overwrites the outputs after `execute_model` returns, as the next CUDA
  graph replay does, and checks that the sent tensors are unchanged. It fails
  without the copy.
- End-to-end PP correctness with CUDA graphs is covered by the existing
  `tests/distributed/test_pipeline_parallel.py`.
- GSM8K and greedy outputs, MiniMax-M3-MXFP8, PP2, main vs this PR.
- Prefill-only serving A/B, MiniMax-M3-MXFP8, PP2.

## Test Results

`pytest tests/v1/worker/test_gpu_worker.py`: 17 passed, 10 skipped (CPU).

### Setup

- One pinned GB300 node (whole node reserved, 2 GPUs used).
- `MiniMaxAI/MiniMax-M3-MXFP8` at revision `c5454eb0`, `-pp 2`.
- CUDA graphs on, prefix caching on, `--max-num-batched-tokens 16384`,
  `--block-size 128`.
- main = merge base `ed1a730795`.
- Four rounds alternating main / PR / PR / main. Each cell below is the mean of
  that arm's two rounds.

### Correctness

| | main | this PR |
| --- | ---: | ---: |
| GSM8K 5-shot, greedy | TODO(results) | TODO(results) |

Greedy outputs (32 prompts x 64 tokens, concurrency 1): TODO(results).

### Prefill-only serving (output length 1)

This targets PP as the prefill role in disaggregated serving. Decode
throughput is not a goal of PP2 here. Throughput is measured at saturation
(offered load above capacity), in computed prompt tokens per second per GPU.
Prefix-cache hits are not counted.

| workload | main | this PR | change |
| --- | ---: | ---: | ---: |
| Shared prefix: one 100K-token prefix + 2-8K new tokens | TODO(results) | TODO(results) | TODO(results) |
| Control: 4-12K fresh tokens | TODO(results) | TODO(results) | TODO(results) |

TTFT at matched offered load, relative to main: TODO(results).

### Where the time goes (synthetic)

This is a microbenchmark of the send path alone: torch.distributed NCCL, stages
running matmul "forwards" of 2-25 ms, outputs written into reused storage, the
engine's in-flight batch limit, and a last stage that syncs every step. It
reports the total time each stage's compute stream was blocked before its
forward or inside its send. Each cell is two runs of 120 steps, about 3 s each.

| scenario | main (wait) | this PR (copy) |
| --- | ---: | ---: |
| PP2, uniform steps, stage 0 | 1317 / 1428 ms | 5 / 5 ms |
| PP4, uniform steps, stages 0-2 | 565-1022 ms | 4-16 ms |
| PP2, varying steps, stage 0 | 87 / 169 ms | 50 / 55 ms |

Synthetic throughput was unchanged in every case.

## Notes

- With TP > 1, each rank copies the whole tensor, although only its 1/TP
  slice is sent. That keeps the code simple, and the cost is one small copy.
- The NCCL send kernel still waits for the next stage's receive on the NCCL
  stream and occupies some SMs while it waits. It no longer blocks compute.
- AI assistance (Claude Code) was used for the implementation, benchmarks and
  this description. The submitter reviewed every changed line and ran the tests
  and benchmarks above.
