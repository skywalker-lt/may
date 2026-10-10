"""Router signal: can the corruption family be read from cheap statistics of a 320-px thumbnail of the (corrupted, letterboxed)
input? 400 val2017 images x {clean, gn/db/ct/br at severities 1,3,5}. Eight features, softmax regression (L2), 5-fold CV by image.
Classes: clean, gn, db, ct, br (family, severity-agnostic). Also saves per-image predicted class for the severity-3 set on the
eval subset so the realised route can be scored from the cross matrix."""
import json, time, numpy as np, cv2
from scipy.optimize import minimize
from common import VAL, GT, OUT, letterbox
from corrupt import apply
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}
E = json.load(open(f'{OUT}/eval_ids.json')) if False else None
ids = sorted(im['id'] for im in gt['images']); rng0 = np.random.default_rng(0); sub400 = [ids[i] for i in rng0.choice(len(ids), 400, replace=False)]
def feats(im_bgr):
    lb, _ = letterbox(im_bgr, 320)  # what a 320 thumbnail router sees (pad 114)
    g = cv2.cvtColor(lb, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    hsv = cv2.cvtColor(lb, cv2.COLOR_BGR2HSV)
    # Immerkaer noise estimate
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    nz = np.abs(cv2.filter2D(g, -1, k)).mean() * np.sqrt(np.pi / 2) / 6
    lap = cv2.Laplacian(g, cv2.CV_32F).var()
    b5 = cv2.GaussianBlur(g, (0, 0), 1.0); b15 = cv2.GaussianBlur(g, (0, 0), 3.0)
    hf = ((g - b5) ** 2).mean(); mf = ((b5 - b15) ** 2).mean()
    gx = cv2.Sobel(b5, cv2.CV_32F, 1, 0); gy = cv2.Sobel(b5, cv2.CV_32F, 0, 1); ge = np.sqrt(gx ** 2 + gy ** 2).mean()
    p1, p99 = np.percentile(g, [1, 99])
    return [g.mean(), g.std(), hsv[:, :, 1].mean() / 255, np.log(lap + 1e-8), np.log(nz + 1e-8), np.log(hf + 1e-10) - np.log(mf + 1e-10), (g > 0.98).mean(), p99 - p1, np.log(ge + 1e-8)]
names = ['lum_mean', 'lum_std', 'sat', 'log_lapvar', 'log_noise', 'log_hf/mf', 'frac_sat', 'range', 'log_grad']
fams = ['clean', 'gn', 'db', 'ct', 'br']; sev = [1, 3, 5]
X, Y, S, IMG = [], [], [], []
t0 = time.time()
for j, iid in enumerate(sub400):
    im = cv2.imread(f'{VAL}/{fn[iid]}'); rng = np.random.default_rng(iid + 7)
    for f in fams:
        for s in (sev if f != 'clean' else [0]):
            X.append(feats(apply(im, f'{f}{s}' if s else 'clean', rng))); Y.append(fams.index(f)); S.append(s); IMG.append(iid)
    if j % 100 == 0: print(j, '%.0fs' % (time.time() - t0), flush=True)
X = np.array(X, np.float64); Y = np.array(Y); S = np.array(S); IMG = np.array(IMG)
np.savez(f'{OUT}/router_feats.npz', X=X, Y=Y, S=S, IMG=IMG, names=names)
mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
K = 5; D = Z.shape[1]
def fit(Zt, Yt, lam=1e-3):
    def f(w):
        Wm = w[:D * K].reshape(D, K); b = w[D * K:]
        L = Zt @ Wm + b; L -= L.max(1, keepdims=True); P = np.exp(L); P /= P.sum(1, keepdims=True)
        loss = -np.log(P[np.arange(len(Yt)), Yt] + 1e-12).mean() + lam * (Wm ** 2).sum()
        G = P.copy(); G[np.arange(len(Yt)), Yt] -= 1; G /= len(Yt)
        return loss, np.concatenate([(Zt.T @ G + 2 * lam * Wm).ravel(), G.sum(0)])
    r = minimize(f, np.zeros(D * K + K), jac=True, method='L-BFGS-B'); return r.x
def pred(w, Zt):
    Wm = w[:D * K].reshape(D, K); b = w[D * K:]; return (Zt @ Wm + b).argmax(1)
uimg = np.array(sub400); folds = np.array_split(np.random.default_rng(1).permutation(uimg), 5)
P = np.zeros(len(Y), int)
for fo in folds:
    te = np.isin(IMG, fo); w = fit(Z[~te], Y[~te]); P[te] = pred(w, Z[te])
print('\n5-fold CV by image, 400 images x 13 conditions; overall accuracy %.3f' % (P == Y).mean())
print('per class (family, all severities):', {fams[k]: round(float((P[Y == k] == k).mean()), 3) for k in range(K)})
for s in sev: print('severity', s, 'accuracy over the four corruptions %.3f' % (P[(S == s)] == Y[(S == s)]).mean(), {fams[k]: round(float((P[(Y == k) & (S == s)] == k).mean()), 3) for k in range(1, K)})
cm = np.zeros((K, K), int)
for y, p in zip(Y, P): cm[y, p] += 1
print('confusion (rows true, cols predicted; clean row has 400 rows, others 1200):'); print('        ' + ' '.join('%6s' % f for f in fams))
for k in range(K): print('%6s  ' % fams[k] + ' '.join('%6d' % v for v in cm[k]))
# severity-3 only accuracy with the CV predictions, and clean
m3 = (S == 3) | (S == 0); print('severity-3 + clean accuracy %.3f' % (P[m3] == Y[m3]).mean())
# single-feature diagnostics
for i, n in enumerate(names): print('%-12s class means (z):' % n, np.round([Z[Y == k, i].mean() for k in range(K)], 2))
json.dump({'img': IMG[m3].tolist(), 'true': Y[m3].tolist(), 'pred': P[m3].tolist(), 'sev': S[m3].tolist(), 'fams': fams}, open(f'{OUT}/router_pred_s3.json', 'w'))
print('done %.0fs' % (time.time() - t0))
