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

# equal-cost heterogeneous M-class oracle: yolo26m (5.36), yolov12m_sdpa (5.53), yolo11m (5.24): selection among
# near-equal models measures complementarity plus selection-on-noise, the right ceiling for candidate A at equal cost.
FILES = {"m": f"{D}/dump_yolo26m_coco.json", "v12": f"{D}/dump_yolov12m_sdpa_coco.json", "v11": f"{D}/dump_yolo11m_coco.json"}
KEYS = list(FILES); byimg = {}; SS = {}
for k, f in FILES.items():
    byimg[k] = {}; res = json.load(open(f))
    for d in res: byimg[k].setdefault(d["image_id"], []).append(d)
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(res), "bbox"); ev.params.imgIds = IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
    s = np.zeros(len(IMG))
    for e in ev.evalImgs:
        if e is None or e["aRng"] != ev.params.areaRng[0]: continue
        sc = np.asarray(e["dtScores"]); keep = sc >= 0.20
        if keep.sum() == 0: continue
        dm = np.asarray(e["dtMatches"])[:, keep]; di = np.asarray(e["dtIgnore"])[:, keep].astype(bool)
        s[pos[e["image_id"]]] += (((dm > 0) & ~di).sum(1) - 0.5 * ((dm == 0) & ~di).sum(1)).mean()
    SS[k] = s; print(f"EVAL4 {k} alone AP={ev.stats[0]:.4f}", flush=True)
def evaluate(choice, tag):
    res = []
    for i, k in zip(IMG, choice): res += byimg[k].get(i, [])
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(res), "bbox"); ev.params.imgIds = IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
    sh = {k: round(float(np.mean(np.array(choice) == k)), 3) for k in KEYS}
    print(f"EVAL4 {tag:40s} AP={ev.stats[0]:.4f} S/M/L={ev.stats[3]:.4f}/{ev.stats[4]:.4f}/{ev.stats[5]:.4f} shares={sh}", flush=True)
c = np.argmax(np.stack([SS[k] for k in KEYS]), 0); evaluate([KEYS[j] for j in c], "oracle best-of m/v12m/11m (equal cost)")
c = np.argmax(np.stack([SS[k] for k in ["m", "v12"]]), 0); evaluate([["m", "v12"][j] for j in c], "oracle best-of m/v12m")
# tie-margin: only switch away from m when the proxy gain exceeds 1.0
c = np.where(SS["v12"] - SS["m"] > 1.0, "v12", "m"); evaluate(list(c), "m, v12m only when proxy gain > 1")
print("DONE4")
