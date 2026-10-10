"""Agent 2 r2: where does M@768 gain over M@640 on the 20% most small-heavy images (agent 3's C proxy)? By object size.
A P2 leaf on 640 features serves objects < 16 px; gains at >= 16 px come from the whole backbone seeing 1.44x pixels."""
import os, sys, json, numpy as np, io, contextlib
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent2')
from mixlib import gt, D
from pycocotools.cocoeval import COCOeval
g = gt(); ids = sorted(g.getImgIds())
z = np.load(f'{D}/val2017_stem_pooled.npz'); assert list(z['image_id']) == ids
nsm = np.array([sum(1 for a in g.imgToAnns[i] if not a['iscrowd'] and a['area'] < 32**2) for i in ids])
def oof(X, y, lam=100., k=5, seed=0):
    rng = np.random.RandomState(seed); fold = rng.randint(0, k, len(y)); p = np.zeros(len(y))
    for f in range(k):
        tr = fold != f; mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Xt = (X[tr] - mu) / sd; Xv = (X[~tr] - mu) / sd
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - y[tr].mean())); p[~tr] = Xv @ w
    return p
rm = oof(z['m640'], np.log1p(nsm)); top = [ids[i] for i in np.argsort(-rm)[:1000]]
rngs = [[0, 1e10], [0, 16**2], [16**2, 32**2], [32**2, 96**2], [96**2, 1e10]]
for nm in ('dumpml_yolo26m_coco', 'dumpml_yolo26m_768_coco'):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = g.loadRes(f'{D}/{nm}.json'); E = COCOeval(g, dt, 'bbox'); E.params.imgIds = top
        E.params.areaRng = rngs; E.params.areaRngLbl = ['all', 'lt16', '16-32', '32-96', 'gt96']; E.evaluate(); E.accumulate()
    pr = E.eval['precision'][:, :, :, :, 2]
    aps = [np.mean(pr[..., a][pr[..., a] > -1]) for a in range(5)]
    print(nm, 'top-20% small-heavy (m640):', ' '.join(f'{l} {v:.4f}' for l, v in zip(E.params.areaRngLbl, aps)), flush=True)
ng = {l: sum(1 for i in top for a in g.imgToAnns[i] if not a['iscrowd'] and lo <= a['area'] < hi) for l, (lo, hi) in zip(['lt16', '16-32', '32-96', 'gt96'], rngs[1:])}
print('GT instances in those images by size:', ng)
