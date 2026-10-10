#!/usr/bin/env python3
"""Round-3 receipt G0 (T4, zero GPU): can the FREE stem feature route M-tail vs L-tail?
Features: moderator's `inputs/dumps/val2017_stem_pooled.npz` (m640 = YOLO26-M pooled layer-5 at 640, 512-d,
already paid inside the engine; n320 = YOLO26-N stem on a 320 thumbnail, 128-d, 0.179 ms).
Target: per-image proxy gain g = proxy_l - proxy_m (101-point per-image AP, categories with GT; proxy.npy).
Routers: ridge regression to g (5-fold CV, never on its own images), fixed quantile: top 42% predicted g -> L.
Also a 1-hidden-layer MLP (numpy, 64 units) with the same CV, and a GT-count rule as the perfect-signal anchor.
Every AP is the exact global pycocotools AP of the chosen mixture of the m / l dumps on full val2017.
Run: OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python stem_router.py > stem_router.log
"""
import pickle, numpy as np
import oracle as O

W = O.W
d = np.load("/data/tmp/ds-yolo/seminar4/inputs/dumps/val2017_stem_pooled.npz")
order = {int(i): k for k, i in enumerate(d["image_id"])}
idx = np.array([order[i] for i in O.imgIds])
P = np.load(f"{W}/proxy.npy")
g = P[:, 3] - P[:, 2]                      # l minus m, per image
ev_m, _ = pickle.load(open(f"{W}/eval_m.pkl", "rb"))
ev_l, _ = pickle.load(open(f"{W}/eval_l.pkl", "rb"))
evs = {"m": ev_m, "l": ev_l}
PL = 0.42
K = int(round(PL * O.NI))
rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(O.NI), 5)


def realised(score, name):
    ch = np.array(["m"] * O.NI, dtype=object)
    ch[np.argsort(-score)[:K]] = "l"
    ap, aps, apm, apl = O.mixed_ap(evs, list(ch))
    corr = np.corrcoef(score, g)[0, 1]
    print(f"ROUTE {name:34s} l_share={K/O.NI:.3f} avg_ms={4.85*(1-PL)+6.06*PL:.2f} AP={ap:.4f} "
          f"S/M/L={aps:.3f}/{apm:.3f}/{apl:.3f} corr(score,gain)={corr:+.3f}", flush=True)
    return ap


def ridge_cv(X, lam):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    out = np.zeros(O.NI)
    for f in folds:
        tr = np.setdiff1d(np.arange(O.NI), f)
        A = np.c_[X[tr], np.ones(len(tr))]
        w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ g[tr])
        out[f] = np.c_[X[f], np.ones(len(f))] @ w
    return out


def mlp_cv(X, H=64, epochs=300, lr=1e-3, wd=1e-3, seed=0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    out = np.zeros(O.NI)
    r = np.random.default_rng(seed)
    for f in folds:
        tr = np.setdiff1d(np.arange(O.NI), f)
        W1 = r.normal(0, 1 / np.sqrt(X.shape[1]), (X.shape[1], H)); b1 = np.zeros(H)
        W2 = r.normal(0, 1 / np.sqrt(H), H); b2 = 0.0
        m = [np.zeros_like(a) for a in (W1, b1, W2)]; v = [np.zeros_like(a) for a in (W1, b1, W2)]
        t = 0
        for ep in range(epochs):
            perm = r.permutation(tr)
            for b in range(0, len(perm), 256):
                i = perm[b:b + 256]; t += 1
                h = np.maximum(X[i] @ W1 + b1, 0); y = h @ W2 + b2
                e = (y - g[i]) / len(i)
                gW2 = h.T @ e; gb2 = e.sum(); gh = np.outer(e, W2) * (h > 0)
                gW1 = X[i].T @ gh + wd * W1; gb1 = gh.sum(0)
                for k, (p, gr) in enumerate(((W1, gW1), (b1, gb1), (W2, gW2))):
                    m[k] = 0.9 * m[k] + 0.1 * gr; v[k] = 0.999 * v[k] + 0.001 * gr * gr
                    p -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
                b2 -= lr * gb2
        out[f] = np.maximum(X[f] @ W1 + b1, 0) @ W2 + b2
    return out


# anchors
realised(g, "oracle-quantile (true gain, top 42%)")
nulls = [realised(np.random.default_rng(s).random(O.NI), f"random null seed {s}") for s in range(3)]
print("NULL mean", np.mean(nulls))
ng = np.array([len(O.gt.getAnnIds(imgIds=[i], iscrowd=None)) for i in O.imgIds])
realised(ng + 1e-3 * rng.random(O.NI), "GT object count (perfect, not free)")
# stem routers
for lam in (100.0, 1000.0, 10000.0):
    realised(ridge_cv(d["m640"][idx], lam), f"ridge m640 lam={lam:g}")
for lam in (30.0, 300.0):
    realised(ridge_cv(d["n320"][idx], lam), f"ridge n320 lam={lam:g}")
realised(mlp_cv(d["m640"][idx]), "mlp64 m640")
# count estimated from the stem feature, then routed by predicted count
cnt_hat = ridge_cv(d["m640"][idx], 1000.0) * 0 + 0  # placeholder shape
X = d["m640"][idx]; Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
for f in folds:
    tr = np.setdiff1d(np.arange(O.NI), f)
    A = np.c_[Xn[tr], np.ones(len(tr))]
    w = np.linalg.solve(A.T @ A + 1000.0 * np.eye(A.shape[1]), A.T @ np.log1p(ng[tr]))
    cnt_hat[f] = np.c_[Xn[f], np.ones(len(f))] @ w
print("CV corr(log count hat, log count) =", np.corrcoef(cnt_hat, np.log1p(ng))[0, 1])
realised(cnt_hat, "ridge m640 -> log count, route by count")
