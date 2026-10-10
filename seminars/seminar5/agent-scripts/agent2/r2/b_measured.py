"""Agent 2 r2: agent 3's branch B measured (untrained) on public YOLO26-M, 400 random val2017 images, CPU fp32.
noP3 = P3 anchors masked before the end2end top-k (exact 'skip the P3 head'); const17 = noP3 + layer 17 replaced by a
per-channel constant (mean over 100 other val images), i.e. P4/P5 lose the P3 bottom-up input as in B.
Compared with agent 3's proxies applied to the same full output: tau16 / tau32 = drop detections under 16 / 32 px."""
import os, sys, json, copy, io, contextlib, numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent2')
from mixlib import gt, D
from pycocotools.cocoeval import COCOeval
g = gt(); sub = json.load(open('m_ids_0_400.json')); S = sorted(sub)
sc = {i: 640.0 / max(g.imgs[i]['width'], g.imgs[i]['height']) for i in g.imgs}
sz = lambda d: np.sqrt(max(d['bbox'][2] * d['bbox'][3], 0)) * sc[d['image_id']]
V = {'dump (TRT fp16)': [d for d in json.load(open(f'{D}/dumpml_yolo26m_coco.json')) if d['image_id'] in set(S)],
     'full (CPU)': json.load(open('m_full_0_400.json')), 'noP3 (measured)': json.load(open('m_noP3_0_400.json')),
     'const17+noP3 (measured B)': json.load(open('m_const17_0_400.json'))}
V['tau16 proxy'] = [d for d in V['full (CPU)'] if sz(d) >= 16]; V['tau32 proxy'] = [d for d in V['full (CPU)'] if sz(d) >= 32]
EV = {}
for k, dts in V.items():
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(g, g.loadRes(dts), 'bbox'); E.params.imgIds = S; E.evaluate(); E.accumulate(); E.summarize()
    EV[k] = E; print(f'{k:28s} AP {E.stats[0]:.4f} S/M/L {E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f}', flush=True)
# router: m640 OOF ridge -> log1p(small count at input scale), ranks within the subset; B share 0.49
ids = sorted(g.getImgIds()); z = np.load(f'{D}/val2017_stem_pooled.npz')
nsm = np.array([sum(1 for a in g.imgToAnns[i] if not a['iscrowd'] and a['area'] * sc[i]**2 < 32**2) for i in ids])
def oof(X, y, lam=100., k=5, seed=0):
    rng = np.random.RandomState(seed); fold = rng.randint(0, k, len(y)); p = np.zeros(len(y))
    for f in range(k):
        tr = fold != f; mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Xt = (X[tr] - mu) / sd; Xv = (X[~tr] - mu) / sd
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - y[tr].mean())); p[~tr] = Xv @ w
    return p
r = dict(zip(ids, oof(z['m640'], np.log1p(nsm)))); I = len(S); K = len(EV['full (CPU)'].params.catIds); A = 4
def mix(base, alt, routed):
    m = np.tile(np.array([i in routed for i in S]), K * A)
    E = COCOeval(g, None, 'bbox'); E.params.imgIds = S; E._paramsEval = copy.deepcopy(E.params)
    E.evalImgs = [a if f else b for b, a, f in zip(EV[base].evalImgs, EV[alt].evalImgs, m)]
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0], E.stats[3]
for sh in (0.3, 0.49):
    routed = set(sorted(S, key=lambda i: r[i])[:int(sh * I)]); rnd = set(list(np.random.RandomState(0).permutation(S))[:int(sh * I)])
    print(f'--- B share {sh} on the 400-image subset (m640 router)')
    for alt in ('tau16 proxy', 'tau32 proxy', 'noP3 (measured)', 'const17+noP3 (measured B)'):
        a, s = mix('full (CPU)', alt, routed); b, _ = mix('full (CPU)', alt, rnd)
        print(f'   A/{alt:28s} routed AP {a:.4f} (AP_S {s:.4f})  random {b:.4f}  vs A alone {EV["full (CPU)"].stats[0]:.4f}: {a-EV["full (CPU)"].stats[0]:+.4f}', flush=True)
