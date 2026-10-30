# PP send buffer: handoff

Delete `pp_handoff/` before opening the upstream PR.

## Branches (gau-nernst/vllm, all rebased on upstream main d14f72c7fe, 2026-10-30)

| Branch | What |
| --- | --- |
| `pp-send-copy` | **PR 1.** Non-last PP ranks snapshot their outputs into one dedicated send buffer, so the next forward no longer waits for the previous send. Four commits (copy design, then the buffer); not squashed on purpose. This doc is the top commit. |
| `pp-send-buffer-eval` | PR 1 (without this doc) + eval-only `VLLM_PP_SEND_BUFFER_DEPTH` knob: 0 = upstream wait-before-forward, N = N buffers. One build serves every A/B arm. |
| `pp-transport-copy-engine` | PR 2 prototype: one-sided send on CUDA copy engines (older preallocated-buffer commit underneath; rebase onto PR 1 when picked up). |
| `pp-transport-rocm-sdma` | ROCm SDMA prototype on top of PR 2. |

`pp-send-copy-eval-base` (upstream + #61294 cherry-pick) was deleted: #61294 is merged.

## Mechanism

- NCCL send is a rendezvous; stage i+1 posts its irecv only after its previous forward.
  Upstream: `start_i(t+1) >= end_{i+1}(t-1)`. With the buffer, the snapshot at the end of
  forward t waits only for the send of t-1.
- It helps only when the slower stage alternates step to step (big batch then small).
  Uniform full batches or a fixed bottleneck stage gain nothing. `pp_sim.py` is an event
  model: alternating sizes 1.77x at PP2, random 1.18x, uniform 1.0x.
- Closed loop at saturation gives uniform batches (no gain). Open loop near the knee is
  where batches vary.

## Results so far (Qwen3.5-397B-A17B-FP8, dummy weights, GB300, prefill-only)

Benchmark: `bench_pp_depth.sh` (100K shared prefix + 2-8K new tokens, output 1, Rust bench).
Capacity probe at 64 in flight, then open-loop Poisson at 0.6/0.75/0.85/0.95/1.0 x C
(depth-0 capacity), reset+prime before each pass, keep pass b; nsys window at 0.85 x C.

- **Stall removed:** stall-behind-send on stage 0, PP2: 7.0% -> 0.0%; PP4 stages 0/1/2:
  10.4/7.2/5.8% -> 0/0/0.3%. Busy time unchanged (~62%).
- **No TTFT gain.** PP2, two independent runs per arm, PR vs upstream mean p50:
  +2.7, -1.3, +1.2, -0.1, -1.2% at 0.6..1.0 x C; same-arm run-to-run spread up to ~7%.
  PP4 first run also flat (knee not reached at 0.95 x C). Depth 2 adds nothing over 1.
- **Capacity:** no regression (PP2 upstream 4.58/4.61, PR 4.55/4.74 req/s).
- The freed time doesn't reach the critical path: stage 1 still starves for input
  (24.6% -> 23.1%). Unexplained: PP4 stage 1 starvation rose 8.8% -> 14.9%.

## Open questions / next steps

1. **Measure on MiniMax-M3.1** (now runnable with PP: #61294 and M3.1 support merged).
   Earlier M3.1 measurements suggested a TTFT gain where Qwen shows none; why is the open
   question (stall on vs off the critical path, real vs random tokens, dummy weights, step
   structure). Copy `qwen35_397b_pp2_send_depth.yaml`, swap model/serve flags (M3 used
   `--trust-remote-code --block-size 128`), keep the bench.
2. Decide PR 1 framing: stall-removal cleanup, TTFT win on M3.1, or fold into PR 2.
3. Update the PR description: `pr_draft_old_copy_design.md` describes the earlier
   copy design and must be rewritten for the single buffer. No absolute TTFT numbers.
4. Real-weight correctness check (GSM8K) and the `pr-checklist` skill before opening.
5. Side question: on stage 0, `ncclDevKernel_Broadcast` was busy ~9 s per 14.6 s window
   (#58542?).

## Files

- `bench_pp_depth.sh <host> <log_dir> <model>`: one arm; `log_dir` must end in `_d<depth>`.
  Depth 0 writes `capacity.txt` next to the arms; `FIXED_CAPACITY=<req/s>` pins it for reruns.
- `qwen35_397b_pp2_send_depth.yaml`: vigil config (fill the `<...>` placeholders); PP4 is the
  same with `-pp 4`, `gpus_per_node: 4`.
- `arm_check.py <arm_dir>`: sanity flags per arm (errors, buffer depth vs arm, failures,
  a/b spread). `analyze_depth.py <run_dir>...`: relative TTFT table + nsys stall per stage.
- `pp_sim.py`: event model above.

Gotchas: vigil pre-creates all variant dirs (key progress on `slurm.out`); cancel with
`scancel <id>`, never Ctrl-C vigil. Same seed across passes replays cached prompts, so always
reset the prefix cache (needs `VLLM_SERVER_DEV_MODE=1`) between passes.
