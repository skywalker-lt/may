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
for models in ["sml", "ml", "nsmlx", "smlx"]:
    lam, c, ml = budget(models, 5.36)
    out[f"shared-stem {models}@5.36"] = evaluate(c, f"A/F shared-stem oracle {models} lam={lam:.2f} L={ml:.2f}")
    # null: same shares, random assignment
    rng = np.random.default_rng(0); cr = list(c); rng.shuffle(cr)
    out[f"shared-stem {models}@5.36 random-null"] = evaluate(cr, f"  null: same shares, random route {models}")
# hardness predictor for F: choose L on the 30% of images where M itself is least confident
maxc = np.zeros(len(IMG)); mid = np.zeros(len(IMG))
for i in IMG:
    sc = np.array([d["score"] for d in byimg["m"].get(i, [])] or [0.0]); maxc[pos[i]] = sc.max(); mid[pos[i]] = ((sc > 0.15) & (sc < 0.5)).sum()
gain = S["l"] - S["m"]; n_l = int(0.3 * len(IMG)); order = np.argsort(-gain)
print(f"m->l proxy gain: mean {gain.mean():.3f}, share of images where l>m {np.mean(gain>0):.3f}, where m>l {np.mean(gain<0):.3f}", flush=True)
for name, key in [("maxconf-low", -maxc), ("midconf-count", mid)]:
    o = np.argsort(-key, kind="stable"); ch = np.array(["m"] * len(IMG), dtype=object); ch[o[:n_l]] = "l"
    ov = len(set(o[:n_l]) & set(order[:n_l])) / n_l
    out[f"F pred m->l 30% {name}"] = evaluate(list(ch), f"F pred m->l p_l=0.30 by m's {name} (overlap {ov:.2f})")
json.dump(out, open(f"{W}/oracle2_results.json", "w"), indent=1); print("DONE2")
