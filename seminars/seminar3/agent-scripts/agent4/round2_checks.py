"""Agent 4, round 2: checks on the moderator's T4 logs and on the other agents' numbers. CPU only, no torch needed.

Run: /data/envs/rtdetr/bin/python /data/tmp/ds-yolo/seminar3/work/agent4/round2_checks.py
"""
import json
import math
import re
from statistics import NormalDist as N

import numpy as np

S3 = "/data/tmp/ds-yolo/seminar3"

# 1. step-0 router balance on the full val2017 (fp32 histogram, moderator section 3)
h = [2323, 376, 1132, 1169]
n = sum(h)
print("shares", [round(x / n, 4) for x in h], "H=%.3f bits" % (-sum(x / n * math.log2(x / n) for x in h)),
      "images/epoch of the smallest expert", round(118287 * min(h) / n))

# 2. engine bytes against 2 bytes x distinct parameters (sizes from batchF2.log)
for nm, p, b in (("if_dense4", 20.411132e6, 45159452), ("wb_top1_if", 41.289980e6, 86812852),
                 ("if_distinct4", 3.580416e6 + 4 * 16.830716e6, 146015756)):
    print(nm, "bytes - 2p = %.2f MB" % ((b - 2 * p) / 1e6), "MiB = %.1f" % (b / 2**20))

# 3. how many flipped-image margins does the log actually hold?
line = [l for l in open(f"{S3}/inputs/receipts_round1_requests/batchF2.log") if "margins of flipped" in l][0]
vals = [float(x) for x in re.findall(r"[0-9.e-]+", line.split("flipped images:")[1]) if re.search(r"\d", x)]
print("flips reported 36; margins printed", len(vals), "max printed", max(vals))
print("expected images with fp32 margin <= 0.0005 at uniform density:", round(0.0592 * 5000 * 0.0005 / 0.01, 1))

# 4. agent 1's MAC total: conv plus attention MatMul
rows = json.load(open(f"{S3}/work/agent1/rows.json"))
print({k: round(sum(r["macs"] for r in rows if r["kind"] == k) / 1e9, 4) for k in {r["kind"] for r in rows}})

# 5. a single-seed cross-run gate: P(pass) and what a pass means. Prior on the true routed-minus-dense delta
#    N(0.001, 0.002) (my estimate); run-to-run sd of one training 0.002 or 0.003 (estimate, unmeasured).
t = np.linspace(-0.01, 0.012, 4401)
pt = np.exp(-0.5 * ((t - 0.001) / 0.002) ** 2)
pt /= pt.sum()
for sd_seed in (0.002, 0.003):
    for thr in (0.002, 0.003):
        pp = np.array([1 - N(x, sd_seed).cdf(thr) for x in t])
        ppass = (pt * pp).sum()
        print(f"sd_seed {sd_seed} gate {thr}: P(pass)={ppass:.2f} P(true>=0.003|pass)={(pt * pp * (t >= 0.003)).sum() / ppass:.2f}")
