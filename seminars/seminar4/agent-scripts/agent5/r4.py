"""Round 4: the two-scale 512/768 mixture re-scored at the shares the last T4 receipt allows (512 branch 3.45-3.49 ms,
768 branch 7.10-7.13 ms, real inputs; 768 share 0.43 at M's real-input 5.05 ms, 0.52 at the 5.36 unset figure),
plus the L4 front fitted from the L4 family table. CPU, 2 threads.
OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python r4.py > r4.log
"""
import numpy as np
from scipy.stats import spearmanr
import oracle as O
from oracle import IMG_IDS, gt, mix, score, MODELS
MODELS.update({"m512": ("dumpml_yolo26m_512_coco.json", 3.781), "m768": ("dumpml_yolo26m_768_coco.json", 6.959)})

# L4 front from the family table (round4/moderator_insert.md, unset-buffer medians) and T4-baselines.md APs
L4 = {"n": 0.917, "s": 1.344, "m": 2.551, "l": 3.346, "x": 6.204}
T4 = {"n": 1.65, "s": 2.75, "m": 5.37, "l": 6.89, "x": 12.41}
AP = {"n": 0.4060, "s": 0.4795, "m": 0.5261, "l": 0.5417, "x": 0.5691}
sl4 = (AP["l"] - AP["m"]) / (L4["l"] - L4["m"]); st4 = (AP["l"] - AP["m"]) / (T4["l"] - T4["m"])
print(f"L4 front: F(L) = 0.5261 + {sl4:.4f} x (L - 2.551)   [T4 slope {st4:.4f}]")
for k in L4: print(f"  {k}: L4/T4 {L4[k]/T4[k]:.3f}; front residual on L4 {AP[k]-(0.5261+sl4*(L4[k]-2.551)):+.4f}")
for name, l4 in [("EsMoE-M sdpa", 4.773), ("M at 768", 3.642), ("M at 512", 2.046), ("if_depth_ml L branch (real)", 3.05),
                 ("M + 4 soft-mixture layers at P4", 2.551 + 4 * (0.136 - 0.018)), ("M + 12 neck blocks x K=4 (est.)", 2.551 + 4.5),
                 ("M + 2 full-attention layers at P4", 2.551 + 0.405 - 2 * 0.018), ("M + batch-4 320 crop pass (est. 0.95x M)", 2.551 * 1.95)]:
    print(f"  bar on L4 at {l4:.3f} ms: {0.5261 + sl4 * (l4 - 2.551) + 0.003:.4f}  ({name})")

d = np.load(f"{O.D}/val2017_stem_pooled.npz"); assert list(d["image_id"]) == IMG_IDS
rng = np.random.RandomState(0); folds = rng.permutation(5000) % 5

def cv_ridge(X, y, lam=10.0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = np.hstack([X, np.ones((len(X), 1))]); pred = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        w = np.linalg.solve(X[tr].T @ X[tr] + lam * np.eye(X.shape[1]), X[tr].T @ y[tr]); pred[te] = X[te] @ w
    return pred

def realised(pred, heavy, light, q, tag):
    thr = np.quantile(pred, 1 - q); ch = np.where(pred >= thr, heavy, light)
    return score(mix(ch), f"{tag}: share {np.mean(ch == heavy):.3f}").stats[0]

count = np.zeros(5000); med_area = np.zeros(5000)
for i, iid in enumerate(IMG_IDS):
    anns = gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False)); count[i] = len(anns)
    med_area[i] = np.median([a["area"] for a in anns]) if anns else 1e6
P = {n: np.load(f"{O.W}/pi_{n}.npy") for n in ["m512", "m768"]}
g = P["m768"] - P["m512"]
pg = cv_ridge(d["m640"], g); pa = cv_ridge(d["n320"], np.log(med_area))
for q in [0.43, 0.52]:
    print(f"== 768 share {q}")
    realised(pg, "m768", "m512", q, "  m640 ridge-on-gain")
    realised(-pa, "m768", "m512", q, "  n320 ridge-on-logarea")
    realised(-np.log(med_area), "m768", "m512", q, "  GT smallest-median-area rule")
    realised(rng.rand(5000), "m768", "m512", q, "  random null")
