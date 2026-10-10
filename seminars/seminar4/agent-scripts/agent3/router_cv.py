"""Round 3, agent 3: a learned router on the pooled layer-5 stem feature (inputs/dumps/val2017_stem_pooled.npz).

Ridge (closed form, 5-fold out-of-fold predictions) from the pooled M-stem feature at 640 (m640, 512-d) and the
N-stem feature at 320 (n320, 128-d) to (a) log1p(GT object count) and (b) the per-image l-minus-m proxy gain.
Images ranked by the out-of-fold prediction; the top p_L go to the L tail; the mixture is scored by pycocotools
on full val2017. Shares p_L = 0.16 and 0.31 (same-session ratio pricing of the conditional M/L engine) and 0.42
(raw 4.85/6.06 pricing).  CPU, 2 threads.
"""
import json, os, sys, numpy as np, contextlib, io, re
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
gt = COCO(f"{D}/instances_val2017.json"); img_ids = sorted(gt.getImgIds()); nI = len(img_ids)
rng = np.random.default_rng(0); TAU = 0.25; idx = {i: k for k, i in enumerate(img_ids)}
src = open(os.path.join(os.path.dirname(__file__), "oracle3.py")).read()
exec(src[src.index("def run_eval"):src.index("\nidx = {i: k")])  # run_eval, per_image from my round-2 script

by_img, QAP, AP = {}, {}, {}
for k, f in [("m", "dump_yolo26m_coco.json"), ("l", "dump_yolo26l_coco.json")]:
    dets = json.load(open(f"{D}/{f}")); ev = run_eval(dets); AP[k] = ev.stats[0]
    q_mine, q_ap, ngt, nsmall = per_image(ev); QAP[k] = q_ap
    d = {}
    for x in dets: d.setdefault(x["image_id"], []).append(x)
    by_img[k] = d
    print(f"{k} AP {AP[k]:.4f}", flush=True)
gain = QAP["l"] - QAP["m"]

z = np.load(f"{D}/val2017_stem_pooled.npz")
assert list(z["image_id"]) == img_ids or sorted(z["image_id"]) == img_ids
order = np.argsort(z["image_id"]); sel = order[np.searchsorted(z["image_id"][order], img_ids)]
feats = {"m640": z["m640"][sel].astype(np.float64), "n320": z["n320"][sel].astype(np.float64)}


def ridge_oof(X, y, alpha, folds=5):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = np.hstack([X, np.ones((len(X), 1))])
    perm = rng.permutation(len(y)); out = np.zeros(len(y))
    for f in range(folds):
        te = perm[f::folds]; tr = np.setdiff1d(perm, te)
        A = X[tr].T @ X[tr] + alpha * np.eye(X.shape[1]); w = np.linalg.solve(A, X[tr].T @ y[tr])
        out[te] = X[te] @ w
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def score(choice):
    return run_eval([d for i, k in zip(img_ids, choice) for d in by_img[k].get(i, [])]).stats[0]


def route(sig, pL):
    n = int(round(pL * nI)); ch = ["m"] * nI
    for j in np.argsort(-sig)[:n]: ch[j] = "l"
    return ch


targets = {"count": np.log1p(ngt), "gain": gain}
preds = {}
for fname, X in feats.items():
    for tname, y in targets.items():
        best = None
        for alpha in (1.0, 10.0, 100.0, 1000.0):
            p = ridge_oof(X, y, alpha); r = spearman(p, y)
            print(f"ridge {fname}->{tname} alpha {alpha:g}: OOF Spearman with target {r:+.3f}, "
                  f"with GT count {spearman(p, ngt):+.3f}, with gain {spearman(p, gain):+.3f}", flush=True)
            if best is None or r > best[0]: best = (r, alpha, p)
        preds[f"{fname}->{tname}"] = best[2]
print(f"GT count vs gain Spearman {spearman(ngt, gain):+.3f}; nsmall vs gain {spearman(nsmall, gain):+.3f}", flush=True)

for pL in (0.16, 0.31, 0.42):
    print(f"\n== L share {pL}: share-null arithmetic {AP['m'] + pL * (AP['l'] - AP['m']):.4f}", flush=True)
    ch0 = route(gain, pL); print(f"proxy-gain route (oracle at this share): {score(ch0):.4f}", flush=True)
    print(f"random route, same share: {score(list(rng.permutation(ch0))):.4f}", flush=True)
    print(f"GT count route: {score(route(ngt + 1e-3 * rng.random(nI), pL)):.4f}", flush=True)
    for name in ("m640->count", "m640->gain", "n320->count"):
        print(f"learned router {name}: {score(route(preds[name], pL)):.4f}", flush=True)
