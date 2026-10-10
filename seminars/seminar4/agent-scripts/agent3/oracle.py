#!/usr/bin/env python3
"""Agent 3 oracle headroom from the per-image val2017 dumps.

Per-image quality proxy q_m(i): mean over the 10 COCO IoU thresholds of TP/(TP+FP+FN) at score >= TAU, computed from
COCOeval.evalImgs (area 'all', maxDet 100) of each model scored alone on the full val set. The mixture under an
average-latency budget B is a greedy knapsack: start from the cheapest model, upgrade images by the best
(q gain / ms) ratio until the mean latency hits B, then score the per-image mixture with pycocotools (full val2017,
the multi-label protocol where available). Upper bound: the dumps are independent models, not shared-stem tails.
"""
import json, os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "2"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
OUT = "/data/tmp/ds-yolo/seminar4/work/agent3"
LAT = {"n": 1.65, "s": 2.75, "m": 5.36, "l": 6.89, "x": 12.41}
FILES = {k: f"{D}/dump_yolo26{k}_coco.json" for k in "nslx"}
FILES["m"] = f"{D}/dumpml_yolo26m_coco.json"
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25

gt = COCO(f"{D}/instances_val2017.json")
img_ids = sorted(gt.getImgIds())
idx = {i: k for k, i in enumerate(img_ids)}
cats = sorted(gt.getCatIds())


def run_eval(dets):
    dt = gt.loadRes(dets)
    ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = img_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev


def per_image_q(ev):
    """q[i] = mean_t TP/(TP+FP+FN) over categories pooled, area all (index 0), maxDet 100 (index 2)."""
    T = len(ev.params.iouThrs); nI = len(img_ids)
    tp = np.zeros((T, nI)); fp = np.zeros((T, nI)); ng = np.zeros(nI)
    K = len(ev.params.catIds); A = len(ev.params.areaRng)
    for k in range(K):
        for i in range(nI):
            e = ev.evalImgs[k * A * nI + 0 * nI + i]
            if e is None: continue
            gi = np.array(e["gtIgnore"], dtype=bool)
            ng[i] += (~gi).sum()
            ds = np.array(e["dtScores"]); keep = ds >= TAU
            if keep.sum() == 0: continue
            dm = np.array(e["dtMatches"])[:, keep]; di = np.array(e["dtIgnore"])[:, keep].astype(bool)
            m = (dm > 0) & ~di
            tp[:, i] += m.sum(1); fp[:, i] += ((dm == 0) & ~di).sum(1)
    fn = ng[None, :] - tp
    den = tp + fp + fn
    q = np.where(den > 0, tp / np.maximum(den, 1), 1.0)  # empty image with no dets: perfect
    return q.mean(0)


dets, q, ap = {}, {}, {}
for k, f in FILES.items():
    dets[k] = json.load(open(f))
    ev = run_eval(dets[k]); ap[k] = ev.stats[0]; q[k] = per_image_q(ev); del ev
    print(f"model {k}: AP {ap[k]:.4f}  mean q {q[k].mean():.4f}  lat {LAT[k]}", flush=True)
by_img = {k: {} for k in FILES}
import gc
for k in FILES:
    for d in dets[k]: by_img[k].setdefault(d["image_id"], []).append(d)
    dets[k] = None; gc.collect()


def score_mix(choice):
    mix = [d for i, k in zip(img_ids, choice) for d in by_img[k].get(i, [])]
    ev = run_eval(mix); return ev.stats[0]


def knapsack(models, budget):
    """Greedy upgrade from the cheapest model by best marginal q-gain per ms until mean latency reaches budget."""
    models = sorted(models, key=lambda k: LAT[k]); nI = len(img_ids)
    choice = [models[0]] * nI; lat = LAT[models[0]]
    cand = []
    for i in range(nI):
        for k in models[1:]:
            g = q[k][i] - q[models[0]][i]; c = LAT[k] - LAT[models[0]]
            if g > 0: cand.append((g / c, g, c, i, k))
    cand.sort(reverse=True); used = set(); total = lat * nI
    for r, g, c, i, k in cand:
        if i in used: continue
        if (total + c) / nI > budget: continue
        used.add(i); choice[i] = k; total += c
    return choice, total / nI


print(f"\nTAU={TAU}. Oracle rows (upper bound; per-image best-of under a mean-latency budget):")
rows = []
for name, models, budget in [
    ("A2 n+x", "nx", 5.36), ("A3 n+m+x", "nmx", 5.36), ("A5 n..x", "nsmlx", 5.36), ("A4 s+m+l+x", "smlx", 5.36),
    ("B casc s->m (s 2.75 + p*5.36)", "sm", 5.36), ("B casc m->x (5.36 + p*12.41) @6.89", "mx", 6.89),
    ("B casc n->m (1.65+p*5.36)", "nm", 5.36), ("A2 m+l @5.36", "ml", 5.36), ("A2 s+l @5.36", "sl", 5.36),
    ("A5 at 2.75 (S latency)", "nsmlx", 2.75), ("A5 at 6.89 (L latency)", "nsmlx", 6.89),
]:
    if name.startswith("B casc"):
        # cascade: cheap always runs, heavy runs on a fraction p; latency = L_cheap + p*L_heavy
        c, h = models[0], models[1]; nI = len(img_ids)
        p_max = (budget - LAT[c]) / LAT[h]
        gain = q[h] - q[c]; order = np.argsort(-gain); nsel = int(p_max * nI)
        choice = [c] * nI
        for j in order[:nsel]:
            if gain[j] > 0: choice[j] = h
        p = sum(1 for x in choice if x == h) / nI; avg = LAT[c] + p * LAT[h]
    else:
        choice, avg = knapsack(models, budget)
    a = score_mix(choice); shares = {k: round(choice.count(k) / len(choice), 3) for k in models}
    rows.append((name, avg, a, shares)); print(f"{name:40s} avg {avg:.2f} ms  AP {a:.4f}  shares {shares}", flush=True)

# Realisable cascade receipt: route on a scalar of the cheap model's output (no oracle).
print("\nCascade on a cheap-model scalar (realisable, not oracle):")
for c, h, budget in [("s", "m", 5.36), ("n", "m", 5.36), ("m", "x", 6.89), ("m", "l", 6.0)]:
    nI = len(img_ids); p_max = (budget - LAT[c]) / LAT[h]; nsel = int(p_max * nI)
    scal = {}
    for i in img_ids:
        ds = sorted([d["score"] for d in by_img[c].get(i, [])], reverse=True)
        scal[i] = (sum(1 for s_ in ds if 0.1 <= s_ < 0.5),   # count of uncertain boxes
                   -(ds[0] if ds else 0.0),                   # low top score
                   -float(np.mean(ds[:10])) if ds else 0.0)   # low mean of top-10
    for si, sname in enumerate(["n_uncertain(0.1-0.5)", "neg_max_score", "neg_mean_top10"]):
        order = sorted(img_ids, key=lambda i: -scal[i][si])
        heavy = set(order[:nsel]); choice = [h if i in heavy else c for i in img_ids]
        avg = LAT[c] + len(heavy) / nI * LAT[h]; a = score_mix(choice)
        g = q[h] - q[c]; r = np.corrcoef([scal[i][si] for i in img_ids], g)[0, 1]
        print(f"  {c}->{h} scalar={sname:22s} p={len(heavy)/nI:.2f} avg {avg:.2f} ms AP {a:.4f} corr(scalar,gain)={r:+.3f}", flush=True)

json.dump({"tau": TAU, "ap_single": ap, "rows": rows}, open(f"{OUT}/oracle_tau{TAU}.json", "w"), indent=1)
