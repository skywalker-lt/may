#!/usr/bin/env python3
"""Round 4 (agent 1): C's learned router re-scored at the shares the last T4 receipt restores (512 branch 3.47, 768 branch 7.115,
M 5.05 real -> 768 share 0.43; at 5.36 unset -> 0.52) and at the L4's est. share (0.21-0.28). Targets: the 768-512 proxy gain and
the small-object count (A2's target), both 5-fold ridge on n320 (the thumbnail feature the two-scale engine computes), out of fold.
Command: OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python router9.py > router9.log"""
import json, os, contextlib, io
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
FILES = {"m512": "dumpml_yolo26m_512_coco.json", "m768": "dumpml_yolo26m_768_coco.json"}
gt = COCO(f"{D}/instances_val2017.json"); IMG = sorted(gt.getImgIds()); pos = {i: k for k, i in enumerate(IMG)}
rng = np.random.default_rng(0)
Z = np.load(f"{D}/val2017_stem_pooled.npz"); zid = Z["image_id"]; order = np.array([pos[int(i)] for i in zid])
X = np.zeros((len(IMG), Z["n320"].shape[1])); X[order] = Z["n320"]
dets, PX = {}, {}
def evaluate(res, tag):
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(res), "bbox"); ev.params.imgIds = IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
    st = ev.stats; print(f"EVAL9 {tag:70s} AP={st[0]:.4f} S/M/L={st[3]:.4f}/{st[4]:.4f}/{st[5]:.4f}", flush=True); return ev, st
def proxy_ap(ev):
    a0 = ev.params.areaRng[0]; per = {}
    for e in ev.evalImgs:
        if e is None or e["aRng"] != a0: continue
        per.setdefault(e["image_id"], []).append(e)
    s = np.zeros(len(IMG)); R = np.linspace(0, 1, 101)
    for img, es in per.items():
        sc = np.concatenate([np.asarray(e["dtScores"]) for e in es]); ngt = sum(int((~np.asarray(e["gtIgnore"]).astype(bool)).sum()) for e in es)
        if len(sc) == 0 or ngt == 0: continue
        dm = np.concatenate([np.asarray(e["dtMatches"]) for e in es], 1); di = np.concatenate([np.asarray(e["dtIgnore"]) for e in es], 1).astype(bool)
        o = np.argsort(-sc); tpm = ((dm > 0) & ~di)[:, o]; fpm = ((dm == 0) & ~di)[:, o]
        ctp = np.cumsum(tpm, 1); cfp = np.cumsum(fpm, 1); rc = ctp / ngt; pr = ctp / np.maximum(ctp + cfp, 1); ap = 0.0
        for t in range(10):
            p = np.maximum.accumulate(pr[t][::-1])[::-1]; idx = np.searchsorted(rc[t], R, side="left"); ap += np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0.0).mean()
        s[pos[img]] = ap / 10
    return s
for k, f in FILES.items():
    r = json.load(open(f"{D}/{f}")); dets[k] = {}
    for d in r: dets[k].setdefault(d["image_id"], []).append(d)
    ev, _ = evaluate(r, f"{k} alone"); PX[k] = proxy_ap(ev)
def mix(choice):
    out = []; keys = ["m512", "m768"]
    for i, c in zip(IMG, choice): out += dets[keys[c]].get(i, [])
    return out
def cv_ridge(X, y, lam=100.0, folds=5):
    n = len(y); idx = rng.permutation(n); pred = np.zeros(n)
    for f in range(folds):
        te = idx[f::folds]; tr = np.setdiff1d(idx, te); mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = (X[tr] - mu) / sd; Xe = (X[te] - mu) / sd; w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - y[tr].mean())); pred[te] = Xe @ w + y[tr].mean()
    return pred
def top(score, share): k = int(round(share * len(IMG))); ch = np.zeros(len(IMG), int); ch[np.argsort(-score)[:k]] = 1; return ch
anns = {i: gt.loadAnns(gt.getAnnIds(imgIds=i, iscrowd=False)) for i in IMG}
nsmall = np.array([sum(a["area"] < 32 * 32 for a in anns[i]) for i in IMG], float)
med = np.array([np.median([a["area"] for a in anns[i]]) if anns[i] else 1e9 for i in IMG])
y = PX["m768"] - PX["m512"]
p_gain = cv_ridge(X, y); p_cnt = cv_ridge(X, np.log1p(nsmall)); p_area = cv_ridge(X, -np.log(med))
from scipy.stats import spearmanr
print("spearman oof: gain-target vs gain %.3f; small-count-target vs gain %.3f (vs nsmall %.3f); area-target vs gain %.3f" % (
    spearmanr(p_gain, y)[0], spearmanr(p_cnt, y)[0], spearmanr(p_cnt, nsmall)[0], spearmanr(p_area, y)[0]))
for share in (0.43, 0.52, 0.25):
    evaluate(mix(top(p_cnt, share)), f"RES learned n320 -> small-object count share={share:.2f}")
    evaluate(mix(top(p_area, share)), f"RES learned n320 -> -log median area share={share:.2f}")
    evaluate(mix(top(p_gain, share)), f"RES learned n320 -> proxy gain share={share:.2f}")
    evaluate(mix(top(nsmall + 1e-3 * rng.random(len(IMG)), share)), f"RES GT small-object count rule share={share:.2f}")
    evaluate(mix(top(-med, share)), f"RES GT smallest-median-area rule share={share:.2f}")
    evaluate(mix((rng.random(len(IMG)) < share).astype(int)), f"RES random null share={share:.2f}")
print("DONE9")
