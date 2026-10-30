# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Event model of PP prefill stages: when does the send buffer help?
Rules (transfer time ignored, it is sub-ms):
- stage i+1 starts batch t after its own batch t-1 and after stage i sent t;
- NCCL send of t from stage i completes when stage i+1 posts the receive,
  i.e. once stage i+1 finished batch t-1;
- depth 0 (upstream): stage i starts t only after its send of t-1 completed;
- depth d>=1: stage i's snapshot of t (end of forward t) waits for its send
  of t-d to complete;
- the scheduler keeps at most PP+1 batches in flight (max_concurrent_batches).
"""

import random


def run(times, pp, depth):
    n = len(times)
    s = [[0.0] * n for _ in range(pp)]
    e = [[0.0] * n for _ in range(pp)]

    def send_done(i, t):  # send of t from stage i completes
        return e[i + 1][t - 1] if t >= 1 else 0.0

    for t in range(n):
        for i in range(pp):
            start = e[i][t - 1] if t else 0.0
            if i:
                start = max(start, e[i - 1][t])
            elif t >= pp + 1:
                start = max(start, e[pp - 1][t - pp - 1])
            if i < pp - 1 and depth == 0 and t >= 1:
                start = max(start, send_done(i, t - 1))
            s[i][t] = start
            end = start + times[t]
            if i < pp - 1 and depth >= 1 and t >= depth:
                end = max(end, send_done(i, t - depth))
            e[i][t] = end
    return e[pp - 1][-1]


random.seed(0)
B, S = 1.0, 0.125  # full 16K batch vs a 2K batch
patterns = {
    "uniform full": [B] * 400,
    "alternating B,S": [B, S] * 200,
    "B,B,S,S": [B, B, S, S] * 100,
    "random 0.1-1.0": [random.uniform(0.1, 1.0) for _ in range(400)],
}
for pp in (2, 4):
    for name, times in patterns.items():
        r = [run(times, pp, d) for d in (0, 1, 2)]
        print(
            f"PP{pp} {name:16s} makespan d0 {r[0]:7.1f}  d1 {r[1]:7.1f} ({r[0] / r[1]:.2f}x)"
            f"  d2 {r[2]:7.1f} ({r[0] / r[2]:.2f}x)  lower bound {sum(times):7.1f}"
        )
