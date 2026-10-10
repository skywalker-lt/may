#!/usr/bin/env python3
"""Agent 5, seminar 4: per-image oracle headroom from the T4 dumps.

Step 1 (cache): per model, run COCOeval once (area 'all', maxDet 100) and derive a per-image AP proxy
(101-point interpolated AP over the image's own detections, averaged over the 10 IoU thresholds and the
categories present; images with no GT get 0 for every model).
Step 2: pick a model per image under an average-latency budget (Lagrangian sweep), build the mixed
result file, score it with the full default pycocotools protocol. Also cascade variants.
"""
import json, os, sys, contextlib, io
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
W = "/data/tmp/ds-yolo/seminar4/work/agent5"
ANN = f"{D}/instances_val2017.json"
MODELS = {  # name: (dump, TRT-native ms from T4-baselines.md)
    "26n": ("dump_yolo26n_coco.json", 1.65), "26s": ("dump_yolo26s_coco.json", 2.75),
    "26m": ("dump_yolo26m_coco.json", 5.36), "26l": ("dump_yolo26l_coco.json", 6.89),
    "26x": ("dump_yolo26x_coco.json", 12.41), "11m": ("dump_yolo11m_coco.json", 5.24),
    "12m": ("dump_yolov12m_sdpa_coco.json", 5.53),
}
gt = COCO(ANN)
IMG_IDS = sorted(gt.getImgIds())
R = np.linspace(0, 1, 101)


def per_image_ap(ev):
    """Per-image AP proxy from ev.evalImgs (areaRng 'all' only)."""
    T = len(ev.params.iouThrs)
    K = len(ev.params.catIds)
    I = len(ev.params.imgIds)
    out = np.zeros(I)
    for i, img in enumerate(ev.params.imgIds):
        scores, matches, dtig, ngt = [], [], [], 0
        for k in range(K):
            e = ev.evalImgs[k * I + i]  # areaRng index 0 == 'all' when it is the only range
            if e is None:
                continue
            gti = np.asarray(e["gtIgnore"], dtype=bool)
            ngt += int((~gti).sum())
            if len(e["dtScores"]):
                scores.append(np.asarray(e["dtScores"]))
                matches.append(np.asarray(e["dtMatches"]))
                dtig.append(np.asarray(e["dtIgnore"], dtype=bool))
        if ngt == 0:
            continue
        if not scores:
            continue
        s = np.concatenate(scores); m = np.concatenate(matches, axis=1); ig = np.concatenate(dtig, axis=1)
        order = np.argsort(-s, kind="mergesort")
        m = m[:, order]; ig = ig[:, order]
        tp = np.logical_and(m > 0, ~ig); fp = np.logical_and(m == 0, ~ig)
        aps = []
        for t in range(T):
            tpc = np.cumsum(tp[t]); fpc = np.cumsum(fp[t])
            rc = tpc / ngt; pr = tpc / np.maximum(tpc + fpc, 1e-9)
            for j in range(len(pr) - 2, -1, -1):
                pr[j] = max(pr[j], pr[j + 1])
            idx = np.searchsorted(rc, R, side="left")
            q = np.array([pr[x] if x < len(pr) else 0.0 for x in idx])
            aps.append(q.mean())
        out[i] = float(np.mean(aps))
    return out


def score(res, tag, full=True):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res)
        ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = IMG_IDS
        if not full:
            ev.params.areaRng = [[0, 1e10]]; ev.params.areaRngLbl = ["all"]
        ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"{tag}: AP={ev.stats[0]:.4f} AP50={ev.stats[1]:.4f} S/M/L={ev.stats[3]:.4f}/{ev.stats[4]:.4f}/{ev.stats[5]:.4f}", flush=True)
    return ev


def load(name):
    return json.load(open(f"{D}/{MODELS[name][0]}"))


def cache():
    for name in MODELS:
        f = f"{W}/pi_{name}.npy"
        if os.path.exists(f):
            continue
        res = load(name)
        ev = score(res, f"single {name}", full=False)
        np.save(f, per_image_ap(ev))
        # cheap-model confidence signals for the cascade question
        sig = {}
        for r in res:
            sig.setdefault(r["image_id"], []).append(r["score"])
        np.save(f"{W}/sig_{name}.npy", np.array([[max(sig.get(i, [0.0])), sum(1 for x in sig.get(i, []) if x > 0.25),
                                                  float(np.mean(sorted(sig.get(i, [0.0]))[-5:]))] for i in IMG_IDS]))


def mix(choice):
    """choice: array of model names per image index -> merged result list."""
    bynm = {}
    for i, nm in enumerate(choice):
        bynm.setdefault(nm, []).append(IMG_IDS[i])
    out = []
    for nm, ids in bynm.items():
        ids = set(ids)
        out += [r for r in load(nm) if r["image_id"] in ids]
    return out


if __name__ == "__main__":
    cache()
    names = sys.argv[1].split(",") if len(sys.argv) > 1 else ["26n", "26s", "26m", "26l", "26x"]
    budgets = [float(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [5.36]
    P = np.stack([np.load(f"{W}/pi_{n}.npy") for n in names], 1)
    L = np.array([MODELS[n][1] for n in names])
    print("per-image proxy means:", dict(zip(names, P.mean(0).round(4))))
    for B in budgets:
        best = None
        for lam in np.concatenate([[0], np.logspace(-4, 0, 200)]):
            c = np.argmax(P - lam * L[None, :] - 1e-9 * L[None, :], 1)
            avgL = L[c].mean()
            if avgL <= B and (best is None or P[np.arange(len(c)), c].sum() > best[0]):
                best = (P[np.arange(len(c)), c].sum(), lam, c, avgL)
        _, lam, c, avgL = best
        shares = {n: round(float((c == i).mean()), 3) for i, n in enumerate(names)}
        ev = score(mix([names[i] for i in c]), f"oracle budget {B} ms: avgL={avgL:.3f} shares={shares} worst={L.max()}")
