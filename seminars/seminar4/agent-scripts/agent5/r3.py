"""Round 3: router receipts on the moderator's pooled stem features (CPU, 2 threads).

OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python r3.py > r3.log
Routers: 5-fold ridge on m640 (512-d, layer-5 of M at 640, free inside the engine) and n320 (128-d, N stem on a
320 thumbnail, 0.179 ms) predicting the per-image gain proxy; top-q predicted gain -> heavy branch; realised AP is
the exact pycocotools AP of the mixture. Shares are set by the measured engine costs (round-3 insert).
"""
import numpy as np
from scipy.stats import spearmanr
import oracle as O
from oracle import IMG_IDS, gt, mix, score, MODELS
MODELS.update({"m512": ("dumpml_yolo26m_512_coco.json", 3.781), "m768": ("dumpml_yolo26m_768_coco.json", 6.959)})

d = np.load(f"{O.D}/val2017_stem_pooled.npz")
assert list(d["image_id"]) == IMG_IDS
rng = np.random.RandomState(0)
folds = rng.permutation(5000) % 5


def cv_ridge(X, y, lam=10.0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    X = np.hstack([X, np.ones((len(X), 1))])
    pred = np.zeros(len(y))
    for k in range(5):
        tr, te = folds != k, folds == k
        A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
        w = np.linalg.solve(A, X[tr].T @ y[tr])
        pred[te] = X[te] @ w
    return pred


def realised(pred, heavy, light, q, tag):
    thr = np.quantile(pred, 1 - q)
    ch = np.where(pred >= thr, heavy, light)
    ev = score(mix(ch), f"{tag}: share {np.mean(ch == heavy):.3f}")
    return ev.stats[0]


def gt_count():
    c = np.zeros(5000); area = np.zeros(5000)
    for i, iid in enumerate(IMG_IDS):
        anns = gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False))
        c[i] = len(anns)
        area[i] = np.median([a["area"] for a in anns]) if anns else 1e6
    return c, area


count, med_area = gt_count()
P = {n: np.load(f"{O.W}/pi_{n}.npy") for n in ["26m", "26l", "m512", "m768"]}

for name, g, heavy, light, shares in [
    ("two-scale 512/768", P["m768"] - P["m512"], "m768", "m512", [0.33, 0.44]),
    ("depth m/l", P["26l"] - P["26m"], "26l", "26m", [0.31, 0.42]),
]:
    print(f"== {name}: Spearman(gain, GT count) {spearmanr(g, count)[0]:+.3f}, (gain, log median area) "
          f"{spearmanr(g, np.log(med_area))[0]:+.3f}")
    for feat in ["m640", "n320"]:
        X = d[feat]
        pg = cv_ridge(X, g)
        pc = cv_ridge(X, np.log1p(count))
        pa = cv_ridge(X, np.log(med_area))
        print(f"  {feat}: CV Spearman pred-gain vs gain {spearmanr(pg, g)[0]:+.3f}; pred-count vs count "
              f"{spearmanr(pc, count)[0]:+.3f}; pred-logarea vs logarea {spearmanr(pa, np.log(med_area))[0]:+.3f}")
        for q in shares:
            realised(pg, heavy, light, q, f"  {feat} ridge-on-gain, share {q}")
        if feat == "n320":
            realised(-pa if heavy == "m768" else pc, heavy, light, shares[0],
                     f"  {feat} ridge-on-{'logarea' if heavy == 'm768' else 'count'}, share {shares[0]}")
    for q in shares:
        rnd = rng.rand(5000)
        realised(rnd, heavy, light, q, f"  random null, share {q}")
    realised(-np.log(med_area) if heavy == "m768" else count, heavy, light, shares[0],
             f"  GT {'smallest median area' if heavy == 'm768' else 'count'} rule, share {shares[0]}")
