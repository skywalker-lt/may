# Routed AP for the pyramid-level branches: GT rule, out-of-fold learned routers, share-matched random nulls.
import numpy as np, json, sys
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent3'); from mix import score, load
W = '/data/tmp/ds-yolo/seminar5/work/agent3/'; D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'
g = np.load(W + 'gt_counts.npz'); ids = g['ids']; C = g['C']; n = len(ids)
R = np.load(W + 'router_scores.npz')
# detection-aware signal (what a router behind P4/P5 heads could see): confident detections >= 32 px at 640 scale
dets = json.load(open(D + 'dumpml_yolo26m_coco.json')); gtj = json.load(open(D + 'instances_val2017.json'))
wh = {i['id']: max(i['width'], i['height']) for i in gtj['images']}
pos = {int(i): j for j, i in enumerate(ids)}; conf_mid = np.zeros(n)
for d in dets:
    s = np.sqrt(d['bbox'][2] * d['bbox'][3]) * 640 / wh[d['image_id']]
    if d['score'] > 0.3 and 32 <= s < 64: conf_mid[pos[d['image_id']]] += 1
from scipy.stats import spearmanr
small = C[:, 0] + C[:, 1]; large = C[:, 3]
print('signal: confident P4-size (32-64 px) detections vs GT small count: Spearman %.3f' % spearmanr(conf_mid, small)[0])
rng = np.random.default_rng(1); tieb = rng.random(n) * 1e-6
def take(key, s):  # lowest key first
    o = np.argsort(key + tieb); return [int(ids[j]) for j in o[:int(round(s * n))]]
tag = 'm'; full = load(tag, 'full')
def table(var, keyname, keys, shares, nulls=3):
    alt = load(tag, var)
    for s in shares:
        row = []
        for kn, k in keys:
            ap = score(full, alt, take(k, s)); row.append(f'{kn} {ap[0]:.4f} (S {ap[1]:.4f})')
        nl = [score(full, alt, [int(x) for x in np.random.default_rng(100 + r).choice(ids, int(round(s * n)), replace=False)])[0] for r in range(nulls)]
        print(f'{var} share {s:.2f}: ' + ' | '.join(row) + f' | null mean {np.mean(nl):.4f} (sd {np.std(nl):.4f})', flush=True)
for var in sys.argv[1].split(','):
    if var.startswith('noP3'):
        keys = [('GT', small.astype(float)), ('m640', R['m640_log1p_small']), ('n320', R['n320_log1p_small']), ('det', np.log1p(conf_mid))]
        table(var, 'small', keys, [0.2, 0.3, 0.4, 0.49, 0.6])
    elif var.startswith('noP5'):
        keys = [('GT', large.astype(float)), ('m640', R['m640_log1p_large']), ('n320', R['n320_log1p_large'])]
        table(var, 'large', keys, [0.1, 0.2, 0.27, 0.35])
