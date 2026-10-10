"""Router receipt on the pooled stem features (MI round 3: inputs/dumps/val2017_stem_pooled.npz).
Command: OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python router_receipt.py > router_receipt.log
Fits 5-fold cross-validated ridge regressors (numpy closed form) on the N-stem-at-320 feature (128-d) and on the
M layer-5 feature at 640 (512-d) for three targets: log1p(GT small-object count), log1p(GT object count), and the
per-image proxy gain of the costly branch. Images are then routed by a fixed quantile of the out-of-fold
prediction at the stated share, and the realised mixture is scored globally with pycocotools. Reuses proxy.npz
(round-2 proxy: matched fraction minus 0.5 FP per GT at score > 0.3).
"""
import json, os, sys, time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
OUT = "/data/tmp/ds-yolo/seminar4/work/agent2"
FILES = {"m": "dumpml_yolo26m_coco.json", "l": "dump_yolo26l_coco.json",
         "m512": "dumpml_yolo26m_512_coco.json", "m768": "dumpml_yolo26m_768_coco.json"}
gt = COCO(os.path.join(D, "instances_val2017.json"))
IMG = sorted(gt.getImgIds()); idx = {i: k for k, i in enumerate(IMG)}; N = len(IMG)
z = np.load(os.path.join(OUT, "proxy.npz")); q = {k: z[k] for k in z.files}
f = np.load(os.path.join(D, "val2017_stem_pooled.npz"))
order = np.array([idx[int(i)] for i in f["image_id"]])
feat = {}
for k in ("n320", "m640"):
    X = np.zeros((N, f[k].shape[1]), dtype=np.float64); X[order] = f[k]
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); feat[k] = X
n_small = np.zeros(N); n_obj = np.zeros(N)
for a in gt.dataset["annotations"]:
    if a.get("iscrowd", 0): continue
    j = idx[a["image_id"]]; n_obj[j] += 1
    if a["area"] < 32 ** 2: n_small[j] += 1
targets = {"log_small": np.log1p(n_small), "log_count": np.log1p(n_obj),
           "gain_768_512": q["m768"] - q["m512"], "gain_l_m": q["l"] - q["m"]}
dets = {k: json.load(open(os.path.join(D, v))) for k, v in FILES.items()}
by_img = {k: {} for k in FILES}
for k in FILES:
    for d in dets[k]: by_img[k].setdefault(d["image_id"], []).append(d)

def score(res, tag):
    dt = gt.loadRes(res); ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = IMG
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"RESULT {tag}: AP={ev.stats[0]:.4f} S/M/L={ev.stats[3]:.4f}/{ev.stats[4]:.4f}/{ev.stats[5]:.4f}", flush=True)
    return ev.stats[0]
def mixture(choice):  # choice[j] is a model key
    res = []
    for j, i in enumerate(IMG): res += by_img[choice[j]].get(i, [])
    return res
def rank(a): return np.argsort(np.argsort(a)).astype(float)
def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])
def cv_ridge(X, y, folds=5, alphas=(1, 10, 100, 1000, 10000)):
    rng = np.random.default_rng(0); perm = rng.permutation(N); best = None
    for al in alphas:
        pred = np.zeros(N)
        for k in range(folds):
            te = perm[k::folds]; tr = np.setdiff1d(perm, te)
            Xt = X[tr]; A = Xt.T @ Xt + al * np.eye(X.shape[1]); w = np.linalg.solve(A, Xt.T @ (y[tr] - y[tr].mean()))
            pred[te] = X[te] @ w + y[tr].mean()
        r = spearman(pred, y)
        if best is None or r > best[0]: best = (r, al, pred)
    return best
def route(pred, costly_share, cheap, costly):
    thr = np.quantile(pred, 1 - costly_share); return [costly if p > thr else cheap for p in pred]

out = {}
print("## cross-validated router quality (Spearman of out-of-fold prediction with the target)")
P = {}
for fk in ("n320", "m640"):
    for tk, y in targets.items():
        r, al, pred = cv_ridge(feat[fk], y); P[(fk, tk)] = pred
        print(f"CV {fk} -> {tk}: spearman={r:.3f} alpha={al}", flush=True)
        out[f"cv_{fk}_{tk}"] = r
print("## oracle-free signals (for reference)")
for tk in ("log_small", "log_count"):
    print(f"GT {tk} vs gain_768_512: {spearman(targets[tk], targets['gain_768_512']):.3f}; vs gain_l_m: {spearman(targets[tk], targets['gain_l_m']):.3f}")

# 1. C, two-scale 512/768. Shares: 0.433 (oracle shares at standalone costs, round 2) and 0.29 (engine-level costs:
# 768 branch 7.14 real = standalone 6.49 + 0.65; 512 branch est. 3.80 + 0.65 = 4.45 unset; (5.36 - 4.45) / (7.62 - 4.45) = 0.29).
rng = np.random.default_rng(1)
for share in (0.433, 0.29):
    tag = f"C512/768 share768={share}"
    for fk, tk in (("n320", "log_small"), ("n320", "gain_768_512"), ("m640", "log_small"), ("m640", "gain_768_512")):
        out[f"{tag} {fk}->{tk}"] = score(mixture(route(P[(fk, tk)], share, "m512", "m768")), f"{tag} router {fk}->{tk}")
    out[f"{tag} GT small"] = score(mixture(route(targets["log_small"] + 1e-3 * targets["log_count"], share, "m512", "m768")), f"{tag} GT small-count rule")
    out[f"{tag} null"] = score(mixture(route(rng.random(N), share, "m512", "m768")), f"{tag} random null")
# 2. shared-stem M/L (4.85 / 6.06 ms): L share 0.42 at 5.36 ms, 0.17 at the twin's 5.05 ms
for share in (0.42, 0.17):
    tag = f"ML shareL={share}"
    for fk, tk in (("m640", "log_count"), ("m640", "gain_l_m"), ("n320", "log_count")):
        out[f"{tag} {fk}->{tk}"] = score(mixture(route(P[(fk, tk)], share, "m", "l")), f"{tag} router {fk}->{tk}")
    out[f"{tag} GT count"] = score(mixture(route(targets["log_count"] + 1e-3 * targets["log_small"], share, "m", "l")), f"{tag} GT count rule")
    out[f"{tag} null"] = score(mixture(route(rng.random(N), share, "m", "l")), f"{tag} random null")
json.dump(out, open(os.path.join(OUT, "router_receipt.json"), "w"), indent=1)
print("DONE")
