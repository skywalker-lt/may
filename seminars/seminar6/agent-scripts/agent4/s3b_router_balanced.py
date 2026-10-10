"""Router, second fit: class-balanced softmax regression on the saved features plus two more (flat-region noise floor,
luminance skew), 5-fold CV by image, and a clean-prior decision rule: route to a corruption set only if its posterior
exceeds tau, else the public set. Reports clean false-alarm rate against corruption recall per severity."""
import json, numpy as np, cv2, time
from scipy.optimize import minimize
from common import VAL, GT, OUT, letterbox
from corrupt import apply
d = np.load(f'{OUT}/router_feats.npz'); X, Y, S, IMG = d['X'], d['Y'], d['S'], d['IMG']; names = list(d['names'])
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}
fams = ['clean', 'gn', 'db', 'ct', 'br']
# two extra features, recomputed in the same order as s3
def extra(im_bgr):
    lb, _ = letterbox(im_bgr, 320); g = cv2.cvtColor(lb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    h, w = g.shape; blk = g[:h // 8 * 8, :w // 8 * 8].reshape(h // 8, 8, w // 8, 8).std(axis=(1, 3))
    core = blk[2:-2, 2:-2]  # avoid the pad
    nf = np.log(np.percentile(core, 5) + 1e-6)
    m = g.mean(); sk = ((g - m) ** 3).mean() / (g.std() ** 3 + 1e-9)
    return [nf, sk]
t0 = time.time(); ex = []; uimg = []
for iid in IMG:
    if not uimg or iid != uimg[-1]: uimg.append(iid)
cache = {}
k = 0
for iid in uimg:
    im = cv2.imread(f'{VAL}/{fn[int(iid)]}'); rng = np.random.default_rng(int(iid) + 7)
    for f in fams:
        for s in ([1, 3, 5] if f != 'clean' else [0]):
            ex.append(extra(apply(im, f'{f}{s}' if s else 'clean', rng))); k += 1
print('extra features %.0fs' % (time.time() - t0), flush=True)
X = np.concatenate([X, np.array(ex)], 1); names += ['log_noisefloor', 'lum_skew']
mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; K = 5; D = Z.shape[1]
wcls = len(Y) / (K * np.bincount(Y, minlength=K)); wt = wcls[Y]
def fit(Zt, Yt, Wt, lam=1e-3):
    def f(w):
        Wm = w[:D * K].reshape(D, K); b = w[D * K:]
        L = Zt @ Wm + b; L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
        loss = -(Wt * np.log(P[np.arange(len(Yt)), Yt] + 1e-12)).sum() / Wt.sum() + lam * (Wm ** 2).sum()
        G = P.copy(); G[np.arange(len(Yt)), Yt] -= 1; G *= Wt[:, None] / Wt.sum()
        return loss, np.concatenate([(Zt.T @ G + 2 * lam * Wm).ravel(), G.sum(0)])
    return minimize(f, np.zeros(D * K + K), jac=True, method='L-BFGS-B').x
def proba(w, Zt):
    Wm = w[:D * K].reshape(D, K); b = w[D * K:]; L = Zt @ Wm + b; L -= L.max(1, keepdims=True); P = np.exp(L); return P / P.sum(1, keepdims=True)
folds = np.array_split(np.random.default_rng(1).permutation(np.array(uimg)), 5)
PR = np.zeros((len(Y), K))
for fo in folds:
    te = np.isin(IMG, fo); w = fit(Z[~te], Y[~te], wt[~te]); PR[te] = proba(w, Z[te])
P = PR.argmax(1)
print('balanced fit, 11 features, 5-fold CV by image: accuracy %.3f' % (P == Y).mean(), {fams[k]: round(float((P[Y == k] == k).mean()), 3) for k in range(K)})
cm = np.zeros((K, K), int)
for y, p in zip(Y, P): cm[y, p] += 1
print('confusion (rows true, cols predicted):'); print('        ' + ' '.join('%6s' % f for f in fams))
for k in range(K): print('%6s  ' % fams[k] + ' '.join('%6d' % v for v in cm[k]))
print('\nclean-prior rule: route to the argmax corruption set only if its posterior > tau, else public')
print('tau   clean false-alarm   corruption recall s1 / s3 / s5   (recall = routed to its own family)')
out = {}
for tau in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
    pc = PR[:, 1:].argmax(1) + 1; pm = PR[np.arange(len(Y)), pc]; dec = np.where(pm > tau, pc, 0); dec = np.where(PR[:, 0] >= pm, 0, dec)
    fa = (dec[Y == 0] != 0).mean(); rec = [(dec[(S == s)] == Y[(S == s)]).mean() for s in (1, 3, 5)]
    print('%.1f   %.3f               %.3f / %.3f / %.3f' % (tau, fa, *rec)); out[tau] = dec
# per-corruption recall at tau 0.7, severity 1
dec = out[0.7]
print('tau 0.7, per family recall by severity:', {fams[k]: [round(float((dec[(Y == k) & (S == s)] == k).mean()), 2) for s in (1, 3, 5)] for k in range(1, K)})
m3 = (S == 3) | (S == 0)
for tau in (0.0, 0.7):
    json.dump({'img': IMG[m3].tolist(), 'true': Y[m3].tolist(), 'pred': out[tau][m3].tolist(), 'sev': S[m3].tolist(), 'fams': fams}, open(f'{OUT}/router_pred_s3_tau{tau}.json', 'w'))
print('done %.0fs' % (time.time() - t0))
