# Router variants for the P3 bit: ridge on concatenated features, logistic on the zero-small target. OOF AUC.
import numpy as np
from scipy.stats import rankdata
from scipy.optimize import minimize
W = '/data/tmp/ds-yolo/seminar5/work/agent3/'; D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'
z = np.load(D + 'val2017_stem_pooled.npz'); g = np.load(W + 'gt_counts.npz'); C = g['C']; n = len(C)
small = C[:, 0] + C[:, 1]; y = (small == 0).astype(float)
fold = np.random.default_rng(0).permutation(n) % 5
def auc(s, pos):
    rk = rankdata(s); return (rk[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())
def std(X): return (X - X.mean(0)) / (X.std(0) + 1e-6)
def logit_oof(X, lam):
    X = np.hstack([std(X), np.ones((n, 1))]); p = np.zeros(n)
    for f in range(5):
        tr = fold != f; Xt, yt = X[tr], y[tr]
        def fg(w):
            m = Xt @ w; l = np.logaddexp(0, m) - yt * m; r = lam * (w[:-1] ** 2).sum()
            gr = Xt.T @ (1 / (1 + np.exp(-m)) - yt); gr[:-1] += 2 * lam * w[:-1]; return l.sum() + r, gr
        w = minimize(fg, np.zeros(X.shape[1]), jac=True, method='L-BFGS-B', options={'maxiter': 500}).x
        p[~tr] = X[~tr] @ w
    return p
out = {}
for name, X in [('m640', z['m640']), ('cat', np.hstack([z['m640'], z['n320']]))]:
    for lam in [10, 100, 1000]:
        p = logit_oof(X.astype(np.float64), lam); a = auc(p, y == 1)
        print(f'logistic {name} lambda {lam}: OOF AUC zero-small {a:.3f}', flush=True); out[f'{name}_{lam}'] = -p
np.savez(W + 'router2_scores.npz', **out)
