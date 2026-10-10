#!/usr/bin/env python3
"""Per-image oracle headroom for input-adaptive mixtures of the T4 dumps.

Method: one full COCOeval.evaluate() per model (matching is per image, so evalImgs entries can be mixed across
models per image and re-accumulated exactly). A per-image proxy AP (COCO 101-point AP over the categories with GT
in that image, averaged over the 10 IoU thresholds, area=all, maxDet=100) is used only to CHOOSE the per-image
model; every reported AP is the exact global pycocotools AP of the mixed evalImgs.
"""
import json, os, sys, pickle, copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
W = "/data/tmp/ds-yolo/seminar4/work/agent4"
LAT = {"n": 1.65, "s": 2.75, "m": 5.36, "l": 6.89, "x": 12.41}  # T4-baselines.md TRT-native ms
MODELS = {k: f"{D}/dump_yolo26{k}_coco.json" for k in "nsmlx"}
gt = COCO(f"{D}/instances_val2017.json")
imgIds = sorted(gt.getImgIds())
catIds = sorted(gt.getCatIds())
NI, NC = len(imgIds), len(catIds)


def run_eval(path):
    dt = gt.loadRes(path)
    ev = COCOeval(gt, dt, "bbox")
    ev.params.imgIds = imgIds
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return ev


def per_image_proxy(ev):
    """proxy[i] = mean over cats-with-GT of 101-pt AP averaged over IoU thresholds, for image i alone.
    Images with no GT get proxy = -0.01 * (#dets with score>0.25) (pure false positives)."""
    T = len(ev.params.iouThrs)
    A = len(ev.params.areaRng)
    rec = np.linspace(0, 1, 101)
    out = np.zeros(NI)
    for i in range(NI):
        aps, fp_only = [], 0
        for c in range(NC):
            e = ev.evalImgs[c * A * NI + 0 * NI + i]  # area index 0 = 'all'
            if e is None:
                continue
            gi = np.array(e["gtIgnore"]).astype(bool)
            npig = int((~gi).sum())
            if npig == 0:
                sc = np.array(e["dtScores"])
                fp_only += int((sc > 0.25).sum())
                continue
            dm = np.array(e["dtMatches"])[:, :100]
            di = np.array(e["dtIgnore"])[:, :100].astype(bool)
            tps = np.logical_and(dm, ~di)
            fps = np.logical_and(~dm.astype(bool), ~di)
            tp = np.cumsum(tps, 1).astype(float)
            fp = np.cumsum(fps, 1).astype(float)
            ap_t = []
            for t in range(T):
                r = tp[t] / npig
                p = tp[t] / np.maximum(tp[t] + fp[t], 1e-9)
                for k in range(len(p) - 2, -1, -1):
                    p[k] = max(p[k], p[k + 1])
                inds = np.searchsorted(r, rec, side="left")
                q = np.array([p[j] if j < len(p) else 0.0 for j in inds])
                ap_t.append(q.mean())
            aps.append(np.mean(ap_t))
        out[i] = np.mean(aps) if aps else -0.01 * fp_only
    return out


def mixed_ap(evs, choice):
    """Exact global AP of the per-image mixture: choice[i] in model keys."""
    base = evs[next(iter(evs))]
    A = len(base.params.areaRng)
    mix = copy.copy(base)
    mix.evalImgs = list(base.evalImgs)
    for i in range(NI):
        src = evs[choice[i]].evalImgs
        for c in range(NC):
            for a in range(A):
                idx = c * A * NI + a * NI + i
                mix.evalImgs[idx] = src[idx]
    mix.eval = {}
    mix.accumulate()
    mix.summarize()
    return mix.stats[0], mix.stats[3], mix.stats[4], mix.stats[5]


if __name__ == "__main__":
    cache = f"{W}/evals.pkl"
    if os.path.exists(cache):
        evs, proxy = pickle.load(open(cache, "rb"))
    else:
        evs, proxy = {}, {}
        for k, p in MODELS.items():
            ck = f"{W}/eval_{k}.pkl"
            if os.path.exists(ck):
                evs[k], proxy[k] = pickle.load(open(ck, "rb"))
                continue
            print("== eval", k, flush=True)
            evs[k] = run_eval(p)
            evs[k].cocoDt = None
            proxy[k] = per_image_proxy(evs[k])
            pickle.dump((evs[k], proxy[k]), open(ck, "wb"))
            print(k, "global AP", evs[k].stats[0], "proxy mean", proxy[k].mean(), flush=True)
        pickle.dump((evs, proxy), open(cache, "wb"))
    keys = list(MODELS)
    P = np.stack([proxy[k] for k in keys], 1)  # NI x 5
    L = np.array([LAT[k] for k in keys])
    np.save(f"{W}/proxy.npy", P)
    print("per-image proxy means:", dict(zip(keys, P.mean(0).round(4))))
    print("fraction of images where each model is the per-image best:", dict(zip(keys, np.bincount(P.argmax(1), minlength=5) / NI)))

    def lagrange(lam, allowed):
        sc = P - lam * L
        sc[:, [j for j in range(5) if keys[j] not in allowed]] = -1e9
        return sc.argmax(1)

    results = []
    for name, allowed, budget in [
        ("unconstrained best-of-5", "nsmlx", None),
        ("best-of-5 at avg 5.36", "nsmlx", 5.36),
        ("n/m/l only at avg 5.36 (candidate A)", "nml", 5.36),
        ("n/m/x only at avg 5.36", "nmx", 5.36),
        ("s/m/l only at avg 5.36", "sml", 5.36),
        ("m/l only at avg 5.36", "ml", 5.36),
        ("n/l only at avg 5.36", "nl", 5.36),
        ("n/x only at avg 5.36", "nx", 5.36),
        ("s/l only at avg 5.36", "sl", 5.36),
        ("s/x only at avg 5.36", "sx", 5.36),
        ("n/s/m only at avg 2.75 (S-latency)", "nsm", 2.75),
        ("n/m only at avg 2.75", "nm", 2.75),
    ]:
        if budget is None:
            ch = lagrange(0.0, allowed)
        else:
            lo, hi = 0.0, 1.0
            for _ in range(60):
                lam = (lo + hi) / 2
                ch = lagrange(lam, allowed)
                if (L[ch]).mean() > budget:
                    lo = lam
                else:
                    hi = lam
            ch = lagrange(hi, allowed)
        avg = L[ch].mean()
        shares = {keys[j]: round(float((ch == j).mean()), 3) for j in range(5) if (ch == j).any()}
        ap, aps, apm, apl = mixed_ap(evs, [keys[j] for j in ch])
        bar = 0.5261 + 0.0102 * (avg - 5.36)
        print(f"ORACLE {name}: avg_lat={avg:.3f} AP={ap:.4f} S/M/L={aps:.3f}/{apm:.3f}/{apl:.3f} front={bar:.4f} margin={ap-bar:+.4f} shares={shares}", flush=True)
        results.append((name, avg, ap, bar, shares))
    json.dump(results, open(f"{W}/oracle_results.json", "w"), indent=1)
