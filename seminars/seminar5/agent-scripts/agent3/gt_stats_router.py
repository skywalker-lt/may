# GT per-image level statistics at the 640 input scale and out-of-fold ridge routers on the pooled features.
import json, numpy as np, os
from scipy.stats import spearmanr
D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'; W = '/data/tmp/ds-yolo/seminar5/work/agent3/'
g = json.load(open(D + 'instances_val2017.json'))
imgs = {i['id']: i for i in g['images']}
z = np.load(D + 'val2017_stem_pooled.npz'); ids = z['image_id']
cnt = {i: np.zeros(4) for i in imgs}  # <16, 16-32, 32-128, >128 at 640 input scale (sqrt area), non-crowd
coco_s = {i: 0 for i in imgs}; coco_l = {i: 0 for i in imgs}
for a in g['annotations']:
    if a['iscrowd']: continue
    i = a['image_id']; s = np.sqrt(a['area']) * 640.0 / max(imgs[i]['width'], imgs[i]['height'])
    cnt[i][0 if s < 16 else 1 if s < 32 else 2 if s <= 128 else 3] += 1
    coco_s[i] += a['area'] < 32 ** 2; coco_l[i] += a['area'] > 96 ** 2
C = np.array([cnt[i] for i in ids]); n = len(ids)
small = C[:, 0] + C[:, 1]; large = C[:, 3]
print('images', n, 'no annotations', int((C.sum(1) == 0).sum()))
print('share with zero objects <32 px (640 scale):', (small == 0).mean().round(3), ' zero <16 px:', (C[:, 0] == 0).mean().round(3))
print('share with zero COCO-small (orig area<32^2):', np.mean([coco_s[i] == 0 for i in ids]).round(3))
print('share with zero objects >128 px:', (large == 0).mean().round(3), ' zero COCO-large:', np.mean([coco_l[i] == 0 for i in ids]).round(3))
print('share with neither <32 nor >128 (P4-only candidates):', ((small == 0) & (large == 0)).mean().round(3))
print('objects per size bin (<16,16-32,32-128,>128):', C.sum(0).astype(int), (C.sum(0) / C.sum()).round(3))
np.savez(W + 'gt_counts.npz', ids=ids, C=C)
# out-of-fold ridge, 5 folds, fixed seed
rng = np.random.default_rng(0); fold = rng.permutation(n) % 5
def oof(X, y, lam):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = np.hstack([X, np.ones((n, 1))]); p = np.zeros(n)
    for f in range(5):
        tr = fold != f; A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1]); A[-1, -1] -= lam
        p[~tr] = X[~tr] @ np.linalg.solve(A, X[tr].T @ y[tr])
    return p
out = {}
for feat in ['n320', 'm640']:
    X = z[feat].astype(np.float64)
    for tgt, y in [('log1p_small', np.log1p(small)), ('log1p_large', np.log1p(large)), ('log1p_all', np.log1p(C.sum(1)))]:
        best = None
        for lam in [1, 10, 100, 1000]:
            p = oof(X, y, lam); r = spearmanr(p, y)[0]
            if best is None or r > best[0]: best = (r, lam, p)
        out[f'{feat}_{tgt}'] = best[2]
        # AUC for "zero" class
        zero = (y == 0); s = -best[2]
        from scipy.stats import rankdata
        rk = rankdata(s); auc = (rk[zero].sum() - zero.sum() * (zero.sum() + 1) / 2) / (zero.sum() * (~zero).sum())
        print(f'router {feat} -> {tgt}: Spearman {best[0]:.3f} (lambda {best[1]}), AUC for zero-count images {auc:.3f}')
np.savez(W + 'router_scores.npz', ids=ids, **out)
