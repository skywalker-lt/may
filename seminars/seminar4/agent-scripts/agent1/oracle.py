#!/usr/bin/env python3
"""Oracle headroom for per-image model mixtures on val2017 from the T4 dumps (agent 1, seminar 4).

Per-image proxy score S[m][img] from one global COCOeval per model (area 'all', maxDet 100):
    S = mean over the 10 IoU thresholds of (TP - 0.5 * FP) over detections with score >= 0.20.
Mixtures are chosen per image by argmax_m (S[m] - lam * L[m]) and then scored GLOBALLY with pycocotools
(multi-label dumps, full val2017). The proxy only chooses; the reported AP is the real metric of the mixture.
A proxy-selected mixture is a lower bound on the true per-image oracle at that budget.
"""
import json, os, sys, time
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
W = "/data/tmp/ds-yolo/seminar4/work/agent1"
LAT = {"n": 1.65, "s": 2.75, "m": 5.36, "l": 6.89, "x": 12.41}  # T4-baselines.md, TRT-native ms
FILES = {k: f"{D}/dump_yolo26{k}_coco.json" for k in LAT}
THR = 0.20

gt = COCO(f"{D}/instances_val2017.json")
IMG = sorted(gt.getImgIds())
pos = {i: k for k, i in enumerate(IMG)}
dets, S, base = {}, {}, {}

def evaluate(res, tag):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
    st = ev.stats
    print(f"EVAL {tag:40s} AP={st[0]:.4f} AP50={st[1]:.4f} S/M/L={st[3]:.4f}/{st[4]:.4f}/{st[5]:.4f}", flush=True)
    return ev, st

for k, f in FILES.items():
    t = time.time(); r = json.load(open(f)); dets[k] = r
    ev, st = evaluate(r, f"yolo26{k} full"); base[k] = st[0]
    s = np.zeros(len(IMG))
    for e in ev.evalImgs:
        if e is None or e["aRng"] != ev.params.areaRng[0]: continue
        sc = np.asarray(e["dtScores"]); keep = sc >= THR
        if keep.sum() == 0: continue
        dm = np.asarray(e["dtMatches"])[:, keep]; di = np.asarray(e["dtIgnore"])[:, keep].astype(bool)
        tp = ((dm > 0) & ~di).sum(1); fp = ((dm == 0) & ~di).sum(1)
        s[pos[e["image_id"]]] += (tp - 0.5 * fp).mean()
    S[k] = s
    print(f"  loaded {k} in {time.time()-t:.0f}s, mean proxy {s.mean():.3f}", flush=True)

np.save(f"{W}/proxy_scores.npy", np.stack([S[k] for k in LAT]))
byimg = {k: {} for k in LAT}
for k in LAT:
    for d in dets[k]: byimg[k].setdefault(d["image_id"], []).append(d)

def mixture(choice):
    out = []
    for i, k in zip(IMG, choice): out += byimg[k].get(i, [])
    return out

def budget_choice(models, budget, lams=np.linspace(0, 3, 61)):
    """largest-score choice whose mean latency <= budget, by a Lagrangian scan"""
    M = np.stack([S[k] for k in models]); L = np.array([LAT[k] for k in models])
    best = None
    for lam in lams:
        c = np.argmax(M - lam * L[:, None], 0); ml = L[c].mean()
        if ml <= budget + 1e-9:
            best = (lam, c, ml, M[c, np.arange(len(IMG))].sum()); break
    return best

results = {}
# 1. unconstrained oracle over n/s/m/l/x
c = np.argmax(np.stack([S[k] for k in LAT]), 0); keys = list(LAT)
ml = np.array([LAT[k] for k in keys])[c].mean()
ev, st = evaluate(mixture([keys[j] for j in c]), f"oracle free, mean L={ml:.2f}")
results["free"] = (st[0], ml, np.bincount(c, minlength=5).tolist())
# 2. budget oracles at YOLO26-M's latency (candidate A) and a little above
for models, budget, tag in [("nsmlx", 5.36, "A oracle nsmlx@5.36"), ("nsml", 5.36, "A oracle nsml@5.36"),
                            ("sml", 5.36, "A oracle sml@5.36"), ("ml", 5.36, "A oracle ml@5.36"),
                            ("nsmlx", 4.0, "A oracle nsmlx@4.0"), ("nsmlx", 2.75, "A oracle nsmlx@2.75")]:
    b = budget_choice(list(models), budget)
    lam, c, ml, _ = b
    ev, st = evaluate(mixture([models[j] for j in c]), f"{tag} lam={lam:.2f} L={ml:.2f}")
    results[tag] = (st[0], ml, np.bincount(c, minlength=len(models)).tolist())
# 3. cascades (candidate B): cheap model always runs, heavy runs on a fraction p; cost = L_cheap + p * L_heavy
for cheap, heavy in [("s", "m"), ("n", "m"), ("n", "l"), ("s", "l")]:
    p = (5.36 - LAT[cheap]) / LAT[heavy]
    n_heavy = int(p * len(IMG))
    gain = S[heavy] - S[cheap]
    order = np.argsort(-gain)
    choice = np.array([cheap] * len(IMG), dtype=object); choice[order[:n_heavy]] = heavy
    ev, st = evaluate(mixture(choice), f"B oracle {cheap}->{heavy} p={p:.2f}")
    results[f"B oracle {cheap}->{heavy}"] = (st[0], 5.36, [len(IMG) - n_heavy, n_heavy])
    # predictor: cheap model's own evidence. (a) low max confidence, (b) count of mid-confidence boxes
    maxc = np.zeros(len(IMG)); mid = np.zeros(len(IMG)); nbox = np.zeros(len(IMG))
    for i in IMG:
        sc = np.array([d["score"] for d in byimg[cheap].get(i, [])] or [0.0])
        maxc[pos[i]] = sc.max(); mid[pos[i]] = ((sc > 0.15) & (sc < 0.5)).sum(); nbox[pos[i]] = (sc >= THR).sum()
    for name, key in [("maxconf-low", -maxc), ("midconf-count", mid), ("nbox", nbox)]:
        o = np.argsort(-key, kind="stable"); choice = np.array([cheap] * len(IMG), dtype=object); choice[o[:n_heavy]] = heavy
        ev, st = evaluate(mixture(choice), f"B pred {cheap}->{heavy} {name}")
        results[f"B pred {cheap}->{heavy} {name}"] = (st[0], 5.36, [len(IMG) - n_heavy, n_heavy])
        # how much of the oracle gain does the predictor capture (overlap of the chosen sets)?
        ov = len(set(o[:n_heavy]) & set(order[:n_heavy])) / max(n_heavy, 1)
        print(f"   overlap with oracle set: {ov:.2f}; spearman-ish corr(gain, key) = {np.corrcoef(gain, key)[0,1]:.3f}", flush=True)
json.dump({k: v for k, v in results.items()}, open(f"{W}/oracle_results.json", "w"), indent=1)
print("BASE", base); print("DONE")
