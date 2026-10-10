"""A within-COCO input-domain check for the router design of lens 6.

Question 1: can a linear probe on the pooled stem features already in the dumps (YOLO26-N@320, 128-d;
YOLO26-M@640 layer-5, 512-d) tell an input domain apart?  The only domain variable visible without new
data is achromatic (grayscale) photographs in val2017, found from the pixels.
Question 2: does the public model's AP differ on that domain (M and L dumps, pycocotools, imgIds restricted)?

One thread, numpy only (no sklearn in the env): ridge-regularised logistic regression by Newton steps,
5-fold cross-validation, balanced accuracy and AUC reported.
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from PIL import Image

IN = "/data/tmp/ds-yolo/seminar6/inputs"
IMG = "/data/datasets/coco/images/val2017"
OUT = "/data/tmp/ds-yolo/seminar6/work/agent6"

t0 = time.time()
d = np.load(f"{IN}/dumps/val2017_stem_pooled.npz")
ids = d["image_id"]
cache = f"{OUT}/gray_ids.json"
if os.path.exists(cache):
    gray = set(json.load(open(cache)))
else:
    gray = set()
    for i, iid in enumerate(ids):
        im = Image.open(f"{IMG}/{int(iid):012d}.jpg")
        if im.mode == "L":
            gray.add(int(iid)); continue
        a = np.asarray(im.convert("RGB").resize((64, 64)), dtype=np.int16)
        if np.abs(a[..., 0] - a[..., 1]).max() <= 2 and np.abs(a[..., 1] - a[..., 2]).max() <= 2:
            gray.add(int(iid))
    json.dump(sorted(gray), open(cache, "w"))
y = np.array([1 if int(i) in gray else 0 for i in ids])
print(f"grayscale images in val2017: {y.sum()} of {len(y)} ({100*y.mean():.2f}%)  [{time.time()-t0:.0f}s]")

def logreg_cv(X, y, lam=1.0, folds=5, seed=0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = np.hstack([X, np.ones((len(X), 1))])
    rng = np.random.default_rng(seed); perm = rng.permutation(len(y))
    p = np.zeros(len(y))
    for f in range(folds):
        te = perm[f::folds]; tr = np.setdiff1d(perm, te)
        w = np.zeros(X.shape[1]); Xt, yt = X[tr], y[tr]
        # class-balanced weights
        sw = np.where(yt == 1, 0.5 / max(yt.sum(), 1), 0.5 / max((1 - yt).sum(), 1)) * len(yt)
        for it in range(25):
            z = Xt @ w; s = 1 / (1 + np.exp(-z))
            g = Xt.T @ (sw * (s - yt)) + lam * w
            H = (Xt * (sw * s * (1 - s))[:, None]).T @ Xt + lam * np.eye(len(w))
            step = np.linalg.solve(H, g); w -= step
            if np.abs(step).max() < 1e-6: break
        p[te] = 1 / (1 + np.exp(-(X[te] @ w)))
    pred = (p > 0.5).astype(int)
    tpr = (pred[y == 1] == 1).mean(); tnr = (pred[y == 0] == 0).mean()
    # AUC by rank statistic
    r = p.argsort().argsort() + 1
    auc = (r[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (1 - y).sum())
    return tpr, tnr, auc, int((pred != y).sum())

for key in ["n320", "m640"]:
    for lam in [1.0, 10.0]:
        tpr, tnr, auc, err = logreg_cv(d[key].astype(np.float64), y, lam=lam)
        print(f"probe on {key} ({d[key].shape[1]}-d), ridge {lam}: 5-fold TPR {tpr:.3f} TNR {tnr:.3f} "
              f"balanced acc {(tpr+tnr)/2:.3f} AUC {auc:.4f} errors {err}/{len(y)}  [{time.time()-t0:.0f}s]")

# Question 2: AP on the two subsets, public M and L at 640 (one-to-one head, multi-label dumps).
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io
gt = COCO(f"{IN}/dumps/instances_val2017.json")
gray_ids = sorted(gray); col_ids = sorted(set(int(i) for i in ids) - gray)
rng = np.random.default_rng(1)
for model, fn in [("M@640", "dumpml_yolo26m_coco.json"), ("L@640", "dump_yolo26l_coco.json")]:
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(f"{IN}/dumps/{fn}")
    res = {}
    for name, sub in [("gray", gray_ids), ("colour", col_ids), ("colour-matched-size", None)]:
        if sub is None:
            # average of 20 random colour subsets of the grayscale subset's size: the sampling noise of the gray number
            vals = []
            for k in range(20):
                s = sorted(rng.choice(col_ids, size=len(gray_ids), replace=False).tolist())
                E = COCOeval(gt, dt, "bbox"); E.params.imgIds = s
                with contextlib.redirect_stdout(io.StringIO()):
                    E.evaluate(); E.accumulate(); E.summarize()
                vals.append(E.stats[0])
            res[name] = (float(np.mean(vals)), float(np.std(vals)))
            continue
        E = COCOeval(gt, dt, "bbox"); E.params.imgIds = sub
        with contextlib.redirect_stdout(io.StringIO()):
            E.evaluate(); E.accumulate(); E.summarize()
        res[name] = (float(E.stats[0]), float(E.stats[1]), float(E.stats[2]), float(E.stats[3]))
    print(f"{model}: gray AP {res['gray'][0]:.4f} (AP50 {res['gray'][1]:.4f}, APs {res['gray'][3]:.4f}) | "
          f"colour AP {res['colour'][0]:.4f} (AP50 {res['colour'][1]:.4f}, APs {res['colour'][3]:.4f}) | "
          f"random colour subsets of the same size: {res['colour-matched-size'][0]:.4f} sd {res['colour-matched-size'][1]:.4f}  [{time.time()-t0:.0f}s]")
