# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""PR tables for the depth sweep: relative TTFT per rate, and stall behind
sends from each arm's nsys window.
Usage: analyze_depth.py <run_dir> [<run_dir> ...]   (vigil run dirs)
"""

import bisect
import glob
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


def load(arm, name):
    f = glob.glob(str(arm / "bench" / name / "*.json"))
    return json.load(open(f[0])) if f else None


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def stalls(arm):
    """Per stage: compute-idle time ending within 200 us of a send ending,
    i.e. the stage resumed because its send completed."""
    rep = glob.glob(str(arm / "worker-0" / "traces" / "*.nsys-rep"))
    if not rep:
        return None
    db = Path(rep[0]).with_suffix(".sqlite")
    if not db.exists():
        subprocess.run(
            ["nsys", "export", "-t", "sqlite", "-f", "true", "-o", str(db), rep[0]],
            check=True,
            capture_output=True,
        )
    c = sqlite3.connect(db)
    names = dict(c.execute("select id, value from StringIds"))
    k = c.execute(
        "select start, end, deviceId, streamId, shortName from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    # Device-to-device copies (the send-buffer snapshot) count as stage activity.
    mc = c.execute(
        "select start, end, deviceId from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 8"
    ).fetchall()
    devs = sorted({d for _, _, d, _, _ in k})
    p2p = {}  # (device, stream) -> sorted end times of SendRecv kernels
    for s, e, d, st, n in k:
        if "SendRecv" in names[n]:
            p2p.setdefault((d, st), []).append(e)
    for v in p2p.values():
        v.sort()

    def matches(ends, ref):
        hit = 0
        for x in ends:
            i = bisect.bisect_left(ref, x - 100_000)
            hit += i < len(ref) and ref[i] <= x + 100_000
        return hit / max(len(ends), 1)

    # Middle stages run receives (from i-1) and sends (to i+1) as SendRecv
    # kernels on separate streams. A stage's receive stream finishes its
    # transfers together with the previous stage's send stream.
    send_streams = {}
    for i, d in enumerate(devs[:-1]):
        own = [key for key in p2p if key[0] == d]
        if i == 0:
            send_streams[d] = own
            continue
        prev = sorted(x for key in send_streams[devs[i - 1]] for x in p2p[key])
        send_streams[d] = [key for key in own if matches(p2p[key], prev) < 0.5]
    res = {}
    for d in devs[:-1]:  # the last stage does not send
        comp = union(
            [
                (s, e)
                for s, e, dd, _, n in k
                if dd == d and "nccl" not in names[n].lower()
            ]
            + [(s, e) for s, e, dd in mc if dd == d]
        )
        send_ends = sorted(x for key in send_streams[d] for x in p2p[key])
        window = comp[-1][1] - comp[0][0]
        idle = stall = 0
        j = 0
        for (s0, e0), (s1, _) in zip(comp, comp[1:]):
            gap = s1 - e0
            if gap < 500_000:
                continue
            idle += gap
            while j < len(send_ends) and send_ends[j] < s1 - 200_000:
                j += 1
            if j < len(send_ends) and send_ends[j] <= s1 + 1_000 and send_ends[j] > e0:
                stall += gap
        res[d] = (stall / window, idle / window)
    return res


for run in map(Path, sys.argv[1:]):
    arms = {int(a.name.rsplit("_d", 1)[1]): a for a in sorted(run.glob("override_*"))}
    base = arms.get(0)
    print(f"== {run.parts[-3]}")
    print("rate x C | p50 TTFT vs depth 0 (d1, d2) | p90 TTFT vs depth 0 (d1, d2)")
    for frac in ["0.6", "0.75", "0.85", "0.95"]:
        b0 = load(base, f"rate_{frac}_b")
        cells50, cells90 = [], []
        for d in (1, 2):
            r = load(arms[d], f"rate_{frac}_b") if d in arms else None
            if b0 and r:
                cells50.append(
                    f"{100 * (r['median_ttft_ms'] / b0['median_ttft_ms'] - 1):+.1f}%"
                )
                cells90.append(
                    f"{100 * (r['p90_ttft_ms'] / b0['p90_ttft_ms'] - 1):+.1f}%"
                )
            else:
                cells50.append("-")
                cells90.append("-")
        print(f"{frac:>8} | {', '.join(cells50):>14} | {', '.join(cells90):>14}")
    for d, arm in sorted(arms.items()):
        cap = load(arm, "capacity")
        st = stalls(arm)
        stx = "; ".join(
            f"stage {i}: stalled behind send {s:.1%} of window (idle {t:.1%})"
            for i, (s, t) in (st or {}).items()
        )
        print(
            f"depth {d}: capacity {cap['steady_state']['request_throughput']:.2f} req/s; {stx}"
            if cap
            else f"depth {d}: -"
        )
