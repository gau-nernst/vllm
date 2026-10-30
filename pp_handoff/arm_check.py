# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sanity checks for one depth-sweep arm; prints a summary and FLAGS."""

import glob
import json
import re
import sys
from pathlib import Path

arm = Path(sys.argv[1])
depth = int(arm.name.rsplit("_d", 1)[1])
flags = []
logs = [
    arm / "slurm.out",
    arm / "hook_post_serve_0_bash.log",
    *arm.glob("worker-0/*.log"),
]
text = "\n".join(p.read_text(errors="ignore") for p in logs if p.exists())
for pat in [
    "OutOfMemory",
    "out of memory",
    "Traceback",
    "Remote job died",
    "AssertionError",
]:
    if pat in text:
        flags.append(f"log contains '{pat}'")
kv = re.findall(r"Available KV cache memory: ([\d.]+) GiB", text)
buf = re.findall(r"PP send buffer: depth (\d+)", text)
if (depth > 0) != bool(buf) or (buf and int(buf[0]) != depth):
    flags.append(f"send buffer log {buf} does not match depth {depth}")
hits = [float(h) for h in re.findall(r"Prefix cache hit rate: ([\d.]+)%", text)]
# The priming request is replayed by design (it caches the shared prefix), so a
# ~100% hit-rate window is expected; replayed passes show up as a/b TTFT gaps.
rates_from = re.findall(r"rates from capacity ([\d.]+)", text)


def load(name):
    f = glob.glob(str(arm / "bench" / name / "*.json"))
    return json.load(open(f[0])) if f else None


cap = load("capacity")
if cap:
    ss = cap.get("steady_state") or {}
    print(
        f"  capacity (64 in flight): {ss.get('request_throughput', 0):.2f} req/s; "
        f"rates from depth-0 capacity {rates_from}"
    )
    if cap["failed"]:
        flags.append(f"capacity: {cap['failed']} failed")
else:
    flags.append("missing capacity pass")
for frac in ["0.6", "0.75", "0.85", "0.95", "1.0"]:
    a, b = load(f"rate_{frac}_a"), load(f"rate_{frac}_b")
    if not (a and b):
        if frac != "1.0" or "rep" in str(arm):
            flags.append(f"missing rate {frac} passes")
        continue
    line = []
    for tag, r in (("a", a), ("b", b)):
        line.append(
            f"{tag}: {r['request_throughput']:.2f} req/s p50 {r['median_ttft_ms']:.0f} "
            f"p90 {r['p90_ttft_ms']:.0f} ms"
        )
        if r["failed"]:
            flags.append(f"rate {frac}{tag}: {r['failed']} failed")
    print(f"  rate {frac:>4} x C  " + " | ".join(line))
    if abs(b["median_ttft_ms"] - a["median_ttft_ms"]) / a["median_ttft_ms"] > 0.15:
        flags.append(f"rate {frac}: p50 TTFT a/b differ >15%")
if not load("profiled"):
    flags.append("missing profiled pass")
if not list(arm.glob("worker-0/traces/*")):
    flags.append("no nsys trace")
print(
    f"  KV GiB {kv}; send buffer depth {buf}; max hit rate {max(hits) if hits else None}"
)
for f in flags:
    print("  FLAG:", f)
