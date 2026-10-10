#!/usr/bin/env python3
"""Round 2 (agent 1): convention check of the per-image oracle, resolution oracle (C) on the new 512/768 dumps,
and the m/l shared-stem oracle at the measured branch costs. Two proxies for the per-image choice:
  'tpfp' = mean over 10 IoU thr of (TP - 0.5 FP) at score >= 0.2  (my round-1 convention)
  'ap'   = per-image 101-point AP (categories pooled, own detections, mean over 10 IoU thr; no-GT images score 0)
           (agent 5's convention, the strongest proxy)
Mixtures are always scored globally with pycocotools on full val2017; nulls are random routes at the same shares.
"""
import json, os, sys, time, contextlib, io
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
FILES = {"n": "dump_yolo26n_coco.json", "s": "dump_yolo26s_coco.json", "m": "dump_yolo26m_coco.json",
         "l": "dump_yolo26l_coco.json", "x": "dump_yolo26x_coco.json",
         "m512": "dumpml_yolo26m_512_coco.json", "m768": "dumpml_yolo26m_768_coco.json",
         "s768": "dumpml_yolo26s_768_coco.json"}
# T4 ms: T4B for n..x; moderator insert section 2 for 512/768 rows; section 3 for shared-stem branches
LAT = {"n": 1.65, "s": 2.75, "m": 5.36, "l": 6.89, "x": 12.41, "mml": 5.355, "m512": 3.781, "m768": 6.959,
       "s768": 3.617, "mT": 4.85, "lT": 6.06}
gt = COCO(f"{D}/instances_val2017.json"); IMG = sorted(gt.getImgIds()); pos = {i: k for k, i in enumerate(IMG)}
rng = np.random.default_rng(0)
dets, P = {}, {"tpfp": {}, "ap": {}}

def evaluate(res, tag):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(res), "bbox"); ev.params.imgIds = IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
    st = ev.stats
    print(f"EVAL5 {tag:58s} AP={st[0]:.4f} S/M/L={st[3]:.4f}/{st[4]:.4f}/{st[5]:.4f}", flush=True)
    return ev, st

def proxies(ev):
    a0 = ev.params.areaRng[0]; per = {}
    for e in ev.evalImgs:
        if e is None or e["aRng"] != a0: continue
        per.setdefault(e["image_id"], []).append(e)
    s1 = np.zeros(len(IMG)); s2 = np.zeros(len(IMG)); R = np.linspace(0, 1, 101)
    for img, es in per.items():
        sc = np.concatenate([np.asarray(e["dtScores"]) for e in es]) if es else np.zeros(0)
        if len(sc) == 0: continue
        dm = np.concatenate([np.asarray(e["dtMatches"]) for e in es], 1)
        di = np.concatenate([np.asarray(e["dtIgnore"]) for e in es], 1).astype(bool)
        ngt = sum(int((~np.asarray(e["gtIgnore"]).astype(bool)).sum()) for e in es)
        keep = sc >= 0.2
        tp = ((dm > 0) & ~di)[:, keep].sum(1); fp = ((dm == 0) & ~di)[:, keep].sum(1)
        s1[pos[img]] = (tp - 0.5 * fp).mean()
        if ngt == 0: continue
        o = np.argsort(-sc); tpm = ((dm > 0) & ~di)[:, o]; fpm = ((dm == 0) & ~di)[:, o]
        ctp = np.cumsum(tpm, 1); cfp = np.cumsum(fpm, 1)
        rc = ctp / ngt; pr = ctp / np.maximum(ctp + cfp, 1)
        ap = 0.0
        for t in range(10):
            p = np.maximum.accumulate(pr[t][::-1])[::-1]
            idx = np.searchsorted(rc[t], R, side="left"); q = np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0.0)
            ap += q.mean()
        s2[pos[img]] = ap / 10
    return s1, s2

for k, f in FILES.items():
    t = time.time(); r = json.load(open(f"{D}/{f}")); dets[k] = {}
    for d in r: dets[k].setdefault(d["image_id"], []).append(d)
    ev, st = evaluate(r, f"{k} alone")
    P["tpfp"][k], P["ap"][k] = proxies(ev); print(f"  {k} loaded {time.time()-t:.0f}s", flush=True)

def mix(choice, keys):
    out = []
    for i, c in zip(IMG, choice): out += dets[keys[c]].get(i, [])
    return out

def oracle(keys, costs, budget, proxy, tag, null=True):
    M = np.stack([P[proxy][k] for k in keys]); L = np.array(costs)
    lo, hi = 0.0, 5.0
    for _ in range(40):
        lam = (lo + hi) / 2; c = np.argmax(M - lam * L[:, None], 0)
        if L[c].mean() <= budget: hi = lam
        else: lo = lam
    c = np.argmax(M - hi * L[:, None], 0); sh = np.bincount(c, minlength=len(keys)) / len(IMG)
    ev, st = evaluate(mix(c, keys), f"{tag} [{proxy}] L={L[c].mean():.2f} shares={np.round(sh,2).tolist()}")
    if null:
        cn = rng.permutation(c); evaluate(mix(cn, keys), f"  null random route, same shares, L={L[cn].mean():.2f}")
    return st[0]

# 1. convention check: n..x at 5.36 with both proxies (differences between agents come from the proxy)
for pr in ("tpfp", "ap"):
    oracle(["n", "s", "m", "l", "x"], [LAT[k] for k in "nsmlx"], 5.36, pr, "A nsmlx@5.36")
# 2. (void: dump_yolo26m_coco.json and dumpml_yolo26m_coco.json are the same file, md5 9fcaf8779c38)
P["tpfp"]["mml"], P["ap"]["mml"] = P["tpfp"]["m"], P["ap"]["m"]; dets["mml"] = dets["m"]
# 3. C: resolution oracle on the real 512/768 dumps, shared costs from the insert (router cost ignored here)
for pr in ("tpfp", "ap"):
    oracle(["m512", "mml", "m768"], [3.781, 5.355, 6.959], 5.355, pr, "C m512/m640/m768@5.355")
    oracle(["m512", "m768"], [3.781, 6.959], 5.355, pr, "C m512/m768@5.355")
    oracle(["m512", "mml", "m768", "s768"], [3.781, 5.355, 6.959, 3.617], 5.355, pr, "C +s768@5.355")
    oracle(["m512", "mml", "m768"], [3.781, 5.355, 6.959], 4.0, pr, "C m512/m640/m768@4.0")
# 4. F/A: m-or-l tails on one stem at the measured branch costs 4.85 / 6.06 (insert section 3)
for pr in ("tpfp", "ap"):
    oracle(["mml", "l"], [4.85, 6.06], 5.36, pr, "F m-tail/l-tail shared stem @5.36")
# 5. stand-in selectors at the affordable L share 0.42: m's own mid-confidence count and m512's count
for name, sig in (("m640 midconf count (0.1<=s<0.5)", None), ("m512 count >=0.25 (thumbnail-like)", None)):
    key = "mml" if name.startswith("m640") else "m512"
    cnt = np.zeros(len(IMG))
    for i in IMG:
        sc = np.array([d["score"] for d in dets[key].get(i, [])])
        cnt[pos[i]] = ((sc >= 0.1) & (sc < 0.5)).sum() if key == "mml" else (sc >= 0.25).sum()
    order = np.argsort(-cnt); c = np.zeros(len(IMG), int); c[order[: int(0.42 * len(IMG))]] = 1
    L = np.array([4.85, 6.06])
    evaluate(mix(c, ["mml", "l"]), f"F stand-in {name}, p_l=0.42, L={L[c].mean():.2f}")
    evaluate(mix(rng.permutation(c), ["mml", "l"]), f"  null random route p_l=0.42")
print("DONE5")
