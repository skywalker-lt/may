"""Agent 2 r2, attack on direction 3: is the level ladder (B = no P3 / A / C = P2 leaf) dominated by a resolution ladder
(512 / 640 / 768) on the same images? Agent 3's own proxies (their evalImgs: noP3_16 = tau16, noP3_32 = tau32; C = M@768).
T4 unset-buffer basis (agent 3's): A 5.36, B 5.36-1.00, C 5.36+1.3 or +1.6; M@512 3.78, M@768 6.97 standalone; the
resolution ladder needs a pre-backbone router (n320 thumbnail, +0.18 ms), the level ladder reads m640 for free."""
import os, sys, pickle, numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent2')
from mixlib import *
A3 = '/data/tmp/ds-yolo/seminar5/work/agent3'
Mf = load('dumpml_yolo26m_coco'); M5 = load('dumpml_yolo26m_512_coco'); M7 = load('dumpml_yolo26m_768_coco')
B16 = pickle.load(open(f'{A3}/ei_m_noP3_16.pkl', 'rb')); B32 = pickle.load(open(f'{A3}/ei_m_noP3_32.pkl', 'rb'))
ids = np.array(Mf['imgIds']); I = len(ids); g = gt(); assert list(B16['imgIds']) == list(ids)
sc = {i: 640.0 / max(g.imgs[i]['width'], g.imgs[i]['height']) for i in g.imgs}
nsm = np.array([sum(1 for a in g.imgToAnns[int(i)] if not a['iscrowd'] and a['area'] * sc[int(i)]**2 < 32**2) for i in ids])
z = np.load(f'{D}/val2017_stem_pooled.npz')
def oof(X, y, lam=100., k=5, seed=0):
    rng = np.random.RandomState(seed); fold = rng.randint(0, k, len(y)); p = np.zeros(len(y))
    for f in range(k):
        tr = fold != f; mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Xt = (X[tr] - mu) / sd; Xv = (X[~tr] - mu) / sd
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - y[tr].mean())); p[~tr] = Xv @ w
    return p
rm = oof(z['m640'], np.log1p(nsm)); rn = oof(z['n320'], np.log1p(nsm))
F = lambda t: 0.5261 + 0.0102 * (t - 5.36)
def route3(score, sB, sC):
    o = np.argsort(score + 1e-9 * np.arange(I)); a = np.ones(I, int); a[o[:int(sB * I)]] = 0
    if sC: a[o[I - int(sC * I):]] = 2
    return a
def rep(tag, caches, a, cost):
    ap, sml = mix_ap(caches, a); t = np.asarray(cost)[a].mean()
    print(f'{tag:62s} AP {ap:.4f} S/M/L {sml[0]:.4f}/{sml[1]:.4f}/{sml[2]:.4f} T4 avg {t:.2f} bar {F(t)+0.003:.4f} AP-bar {ap-F(t)-0.003:+.4f}', flush=True)
print('--- 1-bit, cheap share 0.49')
for nm, Bc in (('B tau16', B16), ('B tau32', B32)):
    rep(f'level: {nm}, m640 router (free)', [Bc, Mf], route3(rm, 0.49, 0), [4.36, 5.36])
rep('resolution: M@512, n320 router (+0.18)', [M5, Mf], route3(rn, 0.49, 0), [3.78 + .18, 5.36 + .18])
rep('resolution: M@512, m640 router (infeasible, info only)', [M5, Mf], route3(rm, 0.49, 0), [3.78, 5.36])
print('--- ladder, cheap 0.49 / expensive 0.20')
for nm, Bc in (('B tau16', B16), ('B tau32', B32)):
    for cC in (1.3, 1.6):
        rep(f'level: {nm} / A / C(=M@768 proxy, +{cC} ms), m640', [Bc, Mf, M7], route3(rm, 0.49, 0.20), [4.36, 5.36, 5.36 + cC])
rep('resolution: 512 / 640 / 768, n320 (+0.18)', [M5, Mf, M7], route3(rn, 0.49, 0.20), [3.96, 5.54, 7.15])
for sB in (0.3, 0.4, 0.5, 0.6, 0.7):
    rep(f'resolution 2-rung shrink share {sB}, n320 (+0.18)', [M5, Mf], route3(rn, sB, 0), [3.96, 5.54])
