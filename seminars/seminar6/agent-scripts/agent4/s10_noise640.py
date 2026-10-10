"""Noise detector on the native-resolution frame (before letterbox; letterbox upsampling of <640 images smooths the noise) instead of a 320 thumbnail: Immerkaer sigma and the 5th-percentile
8x8-block std (flat-region noise floor), single threshold, 400 images x {clean, gn1, gn3, gn5}; also blur/contrast/brightness
at severity 3 as negatives. Reports clean false-alarm vs noise recall."""
import json, numpy as np, cv2, time
from common import VAL, GT, OUT, letterbox
from corrupt import apply
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}
ids = sorted(im['id'] for im in gt['images']); rng0 = np.random.default_rng(0); sub400 = [ids[i] for i in rng0.choice(len(ids), 400, replace=False)]
k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
def f(im):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255  # native resolution (before letterbox), as the preprocessing kernel sees it
    h, w = g.shape; L = np.abs(cv2.filter2D(g, -1, k)); core = L[2:h - 2, 2:w - 2]
    # Immerkaer on the image area only, median-robust variant (median of |L| in 16x16 blocks, 10th percentile)
    hh, ww = core.shape[0] // 16 * 16, core.shape[1] // 16 * 16
    blk = core[:hh, :ww].reshape(hh // 16, 16, ww // 16, 16).mean(axis=(1, 3))
    return [np.log(core.mean() + 1e-6), np.log(np.percentile(blk, 10) + 1e-6)]
conds = ['clean', 'gn1', 'gn3', 'gn5', 'db3', 'ct3', 'br3']; F = {c: [] for c in conds}; t0 = time.time()
for iid in sub400:
    im = cv2.imread(f'{VAL}/{fn[iid]}'); rng = np.random.default_rng(iid + 11)
    for c in conds: F[c].append(f(apply(im, c, rng)))
F = {c: np.array(v) for c, v in F.items()}
print('400 images, %.0fs' % (time.time() - t0))
for j, nm in enumerate(['log mean|L|', 'log p10 block |L|']):
    for fa_target in (0.0025, 0.005, 0.01):
        thr = np.quantile(F['clean'][:, j], 1 - fa_target)
        print('%-18s clean FA %.4f (thr %.3f): recall gn1 %.3f gn3 %.3f gn5 %.3f | db3 %.3f ct3 %.3f br3 %.3f' % (nm, (F['clean'][:, j] > thr).mean(), thr, *[(F[c][:, j] > thr).mean() for c in conds[1:]]))
    print('%-18s max clean %.3f, min gn1 %.3f, min gn3 %.3f' % (nm, F['clean'][:, j].max(), F['gn1'][:, j].min(), F['gn3'][:, j].min()))
