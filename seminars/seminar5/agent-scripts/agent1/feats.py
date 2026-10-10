import numpy as np, mixlib as M
def gt_counts():
    G = M.gt(); ids = sorted(G.getImgIds())
    ns = np.zeros(len(ids)); nm = np.zeros(len(ids)); nl = np.zeros(len(ids)); minA = np.full(len(ids), 1e9)
    for i, im in enumerate(ids):
        for a in G.loadAnns(G.getAnnIds(imgIds=im, iscrowd=False)):
            A = a["area"]; minA[i] = min(minA[i], A)
            if A < 32**2: ns[i] += 1
            elif A < 96**2: nm[i] += 1
            else: nl[i] += 1
    return np.array(ids), ns, nm, nl, minA
def features(ids):
    d = np.load(f"{M.D}/val2017_stem_pooled.npz"); pos = {int(k): j for j, k in enumerate(d["image_id"])}
    j = np.array([pos[int(i)] for i in ids]); return d["n320"][j].astype(np.float64), d["m640"][j].astype(np.float64)
def oof_ridge(X, y, lam=10.0, k=5, seed=0):
    rng = np.random.RandomState(seed); fold = rng.randint(0, k, len(y)); out = np.zeros(len(y))
    for f in range(k):
        tr, te = fold != f, fold == f
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xt = (X[tr] - mu) / sd; Xe = (X[te] - mu) / sd
        ym = y[tr].mean(); w = np.linalg.solve(Xt.T @ Xt + lam * len(Xt) / 100 * np.eye(X.shape[1]), Xt.T @ (y[tr] - ym))
        out[te] = Xe @ w + ym
    return out
