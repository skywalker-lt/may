#!/usr/bin/env python3
"""Round 3 (agent 1): learned routers on the moderator's pooled stem features (val2017_stem_pooled.npz).
(a) m/l shared-stem family: 5-fold ridge on m640 [512-d] and on n320 [128-d] predicting the per-image l-minus-m
    proxy gain; fixed-quantile route (top p_L by predicted gain -> l); realised AP scored globally with pycocotools.
(b) two-scale 512/768: same on n320 predicting the 768-minus-512 gain, at the engine-level 768 share.
Nulls: random route at the same share (seed 0). Command:
  OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python router8.py > router8.log
"""
import json, os, contextlib, io
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
FILES = {"m": "dump_yolo26m_coco.json", "l": "dump_yolo26l_coco.json",
         "m512": "dumpml_yolo26m_512_coco.json", "m768": "dumpml_yolo26m_768_coco.json"}
gt = COCO(f"{D}/instances_val2017.json"); IMG = sorted(gt.getImgIds()); pos = {i: k for k, i in enumerate(IMG)}
rng = np.random.default_rng(0)
Z = np.load(f"{D}/val2017_stem_pooled.npz"); zid = Z["image_id"]; order = np.array([pos[int(i)] for i in zid])
F = {}
for key in ("m640", "n320"):
    X = np.zeros((len(IMG), Z[key].shape[1])); X[order] = Z[key]; F[key] = X
dets, PX = {}, {}
def evaluate(res, tag):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(res), "bbox"); ev.params.imgIds = IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
    st = ev.stats
    print(f"EVAL8 {tag:64s} AP={st[0]:.4f} S/M/L={st[3]:.4f}/{st[4]:.4f}/{st[5]:.4f}", flush=True)
    return ev, st
def proxy_ap(ev):  # per-image 101-point AP, categories pooled (agent 5's convention)
    a0 = ev.params.areaRng[0]; per = {}
    for e in ev.evalImgs:
        if e is None or e["aRng"] != a0: continue
        per.setdefault(e["image_id"], []).append(e)
    s = np.zeros(len(IMG)); R = np.linspace(0, 1, 101)
    for img, es in per.items():
        sc = np.concatenate([np.asarray(e["dtScores"]) for e in es])
        ngt = sum(int((~np.asarray(e["gtIgnore"]).astype(bool)).sum()) for e in es)
        if len(sc) == 0 or ngt == 0: continue
        dm = np.concatenate([np.asarray(e["dtMatches"]) for e in es], 1); di = np.concatenate([np.asarray(e["dtIgnore"]) for e in es], 1).astype(bool)
        o = np.argsort(-sc); tpm = ((dm > 0) & ~di)[:, o]; fpm = ((dm == 0) & ~di)[:, o]
        ctp = np.cumsum(tpm, 1); cfp = np.cumsum(fpm, 1); rc = ctp / ngt; pr = ctp / np.maximum(ctp + cfp, 1); ap = 0.0
        for t in range(10):
            p = np.maximum.accumulate(pr[t][::-1])[::-1]; idx = np.searchsorted(rc[t], R, side="left")
            ap += np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0.0).mean()
        s[pos[img]] = ap / 10
    return s
for k, f in FILES.items():
    r = json.load(open(f"{D}/{f}")); dets[k] = {}
    for d in r: dets[k].setdefault(d["image_id"], []).append(d)
    ev, _ = evaluate(r, f"{k} alone"); PX[k] = proxy_ap(ev)
def mix(choice, keys):
    out = []
    for i, c in zip(IMG, choice): out += dets[keys[c]].get(i, [])
    return out
def cv_ridge(X, y, lam=10.0, folds=5):
    """5-fold ridge on standardised features; returns out-of-fold predictions."""
    n = len(y); idx = rng.permutation(n); pred = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xt = (X[tr] - mu) / sd; Xe = (X[te] - mu) / sd
        A = Xt.T @ Xt + lam * np.eye(X.shape[1]); w = np.linalg.solve(A, Xt.T @ (y[tr] - y[tr].mean()))
        pred[te] = Xe @ w + y[tr].mean()
    return pred
def route_top(score, share, keys, tag):
    k = int(round(share * len(IMG))); ch = np.zeros(len(IMG), int); ch[np.argsort(-score)[:k]] = 1
    evaluate(mix(ch, keys), f"{tag} share={share:.2f}")
def null(share, keys, tag):
    ch = (rng.random(len(IMG)) < share).astype(int); evaluate(mix(ch, keys), f"{tag} random null share={share:.2f}")
# ---- (a) m/l family
y = PX["l"] - PX["m"]
print("corr(y_ml, GT count) =", np.corrcoef(y, np.array([len(gt.getAnnIds(imgIds=i, iscrowd=False)) for i in IMG]))[0, 1])
for feat in ("m640", "n320"):
    for lam in (10.0, 100.0):
        p = cv_ridge(F[feat], y, lam); print(f"ridge {feat} lam={lam}: oof corr with y_ml = {np.corrcoef(p, y)[0,1]:.3f}")
    p = cv_ridge(F[feat], y, 100.0)
    for share in (0.42, 0.33):
        route_top(p, share, ["m", "l"], f"ML ridge router on {feat} (oof)")
for share in (0.42, 0.33): null(share, ["m", "l"], "ML")
cnt = np.array([len(gt.getAnnIds(imgIds=i, iscrowd=False)) for i in IMG]); route_top(cnt, 0.42, ["m", "l"], "ML GT count rule")
# ---- (b) two-scale 512/768
y2 = PX["m768"] - PX["m512"]
for lam in (10.0, 100.0):
    p2 = cv_ridge(F["n320"], y2, lam); print(f"ridge n320 lam={lam}: oof corr with y_768-512 = {np.corrcoef(p2, y2)[0,1]:.3f}")
p2 = cv_ridge(F["n320"], y2, 100.0)
med = np.array([np.median([a["area"] for a in gt.loadAnns(gt.getAnnIds(imgIds=i, iscrowd=False))]) if gt.getAnnIds(imgIds=i, iscrowd=False) else 1e9 for i in IMG])
for share in (0.44, 0.30):
    route_top(p2, share, ["m512", "m768"], "RES ridge router on n320 (oof)")
    route_top(-med, share, ["m512", "m768"], "RES GT smallest-median-area rule")
    null(share, ["m512", "m768"], "RES")
print("DONE8")
