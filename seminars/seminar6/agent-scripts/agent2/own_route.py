"""Feedback route: frame t's scale is chosen from frame t-1's own detections. Still-image proxy: the 'previous frame' is the
same image, seen by the engine that ran last (the low engine: worst-informed case; or the high engine). Temporal
decorrelation q: with prob q the signal comes from a random other image (scene cut / large change)."""
import numpy as np, json, sys
from evlib import *
COST = {"l448": 4.28, "l512": 4.88, "l544": 5.32, "l576": 6.25, "l640": 6.89}
pts = sorted([(3.33,0.4937),(4.88,0.5262),(5.32,0.5329),(6.89,0.5417),(12.41,0.5691)])
def env(t):
    for (x0,y0),(x1,y1) in zip(pts,pts[1:]):
        if x0<=t<=x1: return y0+(y1-y0)*(t-x0)/(x1-x0)
names = ["l448","l512","l544","l576","l640"]; S = {n: evaluated(n) for n in names}
ids, cnt, small, med = gt_stats(); I = len(ids); pos = {i:k for k,i in enumerate(ids)}
def own(name, thr):
    c = np.zeros(I); cs = np.zeros(I)
    for d in json.load(open(PATHS[name])):
        if d["score"] > thr:
            k = pos[d["image_id"]]; c[k] += 1
            if d["bbox"][2]*d["bbox"][3] < 32**2: cs[k] += 1
    return c, cs
sig = {}
for n in ["l448","l640"]:
    for thr in [0.25, 0.4]:
        c, cs = own(n, thr); sig[f"own{n[1:]}_c{thr}"] = c; sig[f"own{n[1:]}_s{thr}"] = cs + 1e-3*c
sig["gt_count"] = cnt; sig["gt_small"] = small + 1e-3*cnt
for k in ["own448_c0.25","own448_s0.25","own640_c0.25"]:
    print("rank corr with gt_count", k, round(float(np.corrcoef(np.argsort(np.argsort(sig[k])), np.argsort(np.argsort(cnt)))[0,1]),3))
rng = np.random.default_rng(0)
def run(lo, hi, p, label, rule, reps=1):
    k = int(round(p*I)); aps = []
    for r in range(reps):
        rr = rule + 1e-4*rng.random(I); ch = np.zeros(I,int); ch[np.argsort(-rr, kind="stable")[:k]] = 1
        aps.append(score([S[lo],S[hi]], ch)[0])
    ap = float(np.mean(aps)); t = p*COST[hi]+(1-p)*COST[lo]; sd = float(np.std(aps)) if reps>1 else 0
    print(f"{lo}/{hi} p={p:.2f} {label:16s} AP={ap:.4f} sd={sd:.4f} T4={t:.2f} env={env(t):.4f} m_env={ap-env(t):+.4f}", flush=True)
    return ap
pairs = [("l448","l640"),("l512","l640"),("l448","l576")] if len(sys.argv) < 2 else [tuple(sys.argv[1].split(","))]
for lo, hi in pairs:
    for p in [0.3, 0.5, 0.7]:
        run(lo, hi, p, "random", rng.random(I), reps=3)
        for k in ["own448_c0.25","own448_s0.25","own448_c0.4","own640_c0.25","own640_s0.25","gt_count","gt_small"]:
            run(lo, hi, p, k, sig[k])
        for q in [0.1, 0.3]:
            s = sig["own448_s0.25"].copy(); m = rng.random(I) < q; s[m] = s[rng.integers(0, I, m.sum())]
            run(lo, hi, p, f"own448_s q={q}", s)
