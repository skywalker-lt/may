#!/usr/bin/env python3
"""Shared-stem budgets (candidates A and F) using the proxies saved by oracle.py. Tail costs (ms, estimate from
the split of receipts/t4/profile_wb_top1_if.txt): stem+post 2.71; tails S 1.0, M 2.40, L 3.6, X 7.6."""
import json, os, contextlib, io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"; W = "/data/tmp/ds-yolo/seminar4/work/agent1"
KEYS = "nsmlx"; TAIL = {"n": 0.5, "s": 1.0, "m": 2.40, "l": 3.6, "x": 7.6}; STEM = 2.71
gt = COCO(f"{D}/instances_val2017.json"); IMG = sorted(gt.getImgIds()); pos = {i: k for k, i in enumerate(IMG)}
P = np.load(f"{W}/proxy_scores.npy"); S = {k: P[j] for j, k in enumerate(KEYS)}
byimg = {}
for k in KEYS:
    byimg[k] = {}
    for d in json.load(open(f"{D}/dump_yolo26{k}_coco.json")): byimg[k].setdefault(d["image_id"], []).append(d)
def evaluate(choice, tag):
    res = []
    for i, k in zip(IMG, choice): res += byimg[k].get(i, [])
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(res), "bbox"); ev.params.imgIds = IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
    sh = {k: round(float(np.mean(np.array(choice) == k)), 3) for k in KEYS if k in set(choice)}
    print(f"EVAL2 {tag:46s} AP={ev.stats[0]:.4f} S/M/L={ev.stats[3]:.4f}/{ev.stats[4]:.4f}/{ev.stats[5]:.4f} shares={sh}", flush=True)
    return ev.stats[0]
def budget(models, mean_total):
    M = np.stack([S[k] for k in models]); L = np.array([TAIL[k] for k in models])
    for lam in np.linspace(0, 4, 161):
        c = np.argmax(M - lam * L[:, None], 0)
        if STEM + L[c].mean() <= mean_total + 1e-9: return lam, [models[j] for j in c], STEM + L[c].mean()

out = {}
LAT = {"n": 1.65, "s": 2.75, "m": 5.36, "l": 6.89, "x": 12.41}
for cheap, heavy in [("s", "m"), ("n", "m"), ("n", "l"), ("s", "l")]:
    p = (5.36 - LAT[cheap]) / LAT[heavy]; n_h = int(p * len(IMG))
    gain = S[heavy] - S[cheap]; order = np.argsort(-gain)
    ch = np.array([cheap] * len(IMG), dtype=object); ch[order[:n_h]] = heavy
    out[f"B oracle {cheap}->{heavy}"] = evaluate(list(ch), f"B oracle {cheap}->{heavy} p={p:.2f} (independent costs)")
    maxc = np.zeros(len(IMG)); mid = np.zeros(len(IMG))
    for i in IMG:
        sc = np.array([d["score"] for d in byimg[cheap].get(i, [])] or [0.0]); maxc[pos[i]] = sc.max(); mid[pos[i]] = ((sc > 0.15) & (sc < 0.5)).sum()
    for name, key in [("maxconf-low", -maxc), ("midconf-count", mid)]:
        o = np.argsort(-key, kind="stable"); ch = np.array([cheap] * len(IMG), dtype=object); ch[o[:n_h]] = heavy
        ov = len(set(o[:n_h]) & set(order[:n_h])) / max(n_h, 1)
        out[f"B pred {cheap}->{heavy} {name}"] = evaluate(list(ch), f"B pred {cheap}->{heavy} by {name} (overlap {ov:.2f}, corr {np.corrcoef(gain, key)[0,1]:.2f})")
json.dump(out, open(f"{W}/oracle3_results.json", "w"), indent=1); print("DONE3")
