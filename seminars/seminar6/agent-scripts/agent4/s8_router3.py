"""Three-way route for the revised construct: public | noise stem set | contrast stem set. Same 5,200 thumbnails as s3
(400 images x clean + 4 families x severities 1/3/5), 11 statistics (s3b's extra two recomputed and saved here),
5-fold CV by image, class-balanced softmax over the 5 families, then decision: noise if P(gn) > t_gn, contrast if
P(ct) > t_ct, else public. Expected clean-image tax = FA_gn x 0.218 + FA_ct x 0.051 (the measured k21 misroute taxes)."""
import json, numpy as np, cv2, time, os
from scipy.optimize import minimize
from common import VAL, GT, OUT, letterbox
from corrupt import apply
d = np.load(f'{OUT}/router_feats.npz'); X, Y, S, IMG = d['X'], d['Y'], d['S'], d['IMG']
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}; fams = ['clean', 'gn', 'db', 'ct', 'br']
p = f'{OUT}/router_feats_extra.npy'
if os.path.exists(p): ex = np.load(p)
else:
    def extra(im_bgr):
        lb, _ = letterbox(im_bgr, 320); g = cv2.cvtColor(lb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        h, w = g.shape; blk = g[:h // 8 * 8, :w // 8 * 8].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3)); core = blk[2:-2, 2:-2]
        m = g.mean(); return [np.log(np.percentile(core, 5) + 1e-6), ((g - m) ** 3).mean() / (g.std() ** 3 + 1e-9)]
    ex = []; uimg = list(dict.fromkeys(IMG.tolist()))
    for iid in uimg:
        im = cv2.imread(f'{VAL}/{fn[int(iid)]}'); rng = np.random.default_rng(int(iid) + 7)
        for f in fams:
            for s in ([1, 3, 5] if f != 'clean' else [0]): ex.append(extra(apply(im, f'{f}{s}' if s else 'clean', rng)))
    ex = np.array(ex); np.save(p, ex)
X = np.concatenate([X, ex], 1); mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; K = 5; D = Z.shape[1]
wt = (len(Y) / (K * np.bincount(Y, minlength=K)))[Y]
def fit(Zt, Yt, Wt, lam=1e-3):
    def f(w):
        Wm = w[:D * K].reshape(D, K); b = w[D * K:]; L = Zt @ Wm + b; L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
        G = P.copy(); G[np.arange(len(Yt)), Yt] -= 1; G *= Wt[:, None] / Wt.sum()
        return -(Wt * np.log(P[np.arange(len(Yt)), Yt] + 1e-12)).sum() / Wt.sum() + lam * (Wm ** 2).sum(), np.concatenate([(Zt.T @ G + 2 * lam * Wm).ravel(), G.sum(0)])
    return minimize(f, np.zeros(D * K + K), jac=True, method='L-BFGS-B').x
def proba(w, Zt):
    L = Zt @ w[:D * K].reshape(D, K) + w[D * K:]; L -= L.max(1, keepdims=True); P = np.exp(L); return P / P.sum(1, keepdims=True)
uimg = np.array(list(dict.fromkeys(IMG.tolist()))); folds = np.array_split(np.random.default_rng(1).permutation(uimg), 5); PR = np.zeros((len(Y), K))
for fo in folds:
    te = np.isin(IMG, fo); PR[te] = proba(fit(Z[~te], Y[~te], wt[~te]), Z[te])
cl = Y == 0
print('t_gn  t_ct | clean->noise  clean->contrast  exp. clean tax | noise recall s1/s3/s5 | contrast recall s1/s3/s5 | blur->noise  bright->noise')
for tg in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
    for tc in (0.5, 0.9):
        dg = PR[:, 1] > tg; dc = (PR[:, 3] > tc) & ~dg
        fg, fc = dg[cl].mean(), dc[cl].mean()
        rg = [dg[(Y == 1) & (S == s)].mean() for s in (1, 3, 5)]; rc = [dc[(Y == 3) & (S == s)].mean() for s in (1, 3, 5)]
        print('%.2f  %.2f | %.4f        %.4f           %+.4f        | %.2f / %.2f / %.2f      | %.2f / %.2f / %.2f         | %.3f        %.3f' % (tg, tc, fg, fc, -(fg * 0.218 + fc * 0.051), *rg, *rc, dg[Y == 2].mean(), dg[Y == 4].mean()))
