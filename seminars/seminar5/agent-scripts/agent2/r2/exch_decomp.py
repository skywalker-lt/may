"""Agent 2 r2: decompose agent 7's exchange route {M@512, M@640, L@640} into its resolution decision and its depth decision.
Same shares as agent 7 (q512 = 0.50, qL = 0.399); each decision either count-routed (n320 OOF ridge -> log1p count) or random.
Also: AP of M@512 vs M@640 by GT-count bucket (is the resolution loss density-dependent per instance, unlike depth?)."""
import os, sys, numpy as np, copy, io, contextlib
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent2')
from mixlib import *
from pycocotools.cocoeval import COCOeval
M5 = load('dumpml_yolo26m_512_coco'); M = load('dumpml_yolo26m_coco'); L = load('dump_yolo26l_coco')
ids = np.array(M['imgIds']); I = len(ids); g = gt()
ngt = np.array([sum(1 for a in g.imgToAnns[int(i)] if not a['iscrowd']) for i in ids])
z = np.load(f'{D}/val2017_stem_pooled.npz'); assert (z['image_id'] == ids).all()
def oof(X, y, lam=100., k=5, seed=0):
    rng = np.random.RandomState(seed); fold = rng.randint(0, k, len(y)); p = np.zeros(len(y))
    for f in range(k):
        tr = fold != f; mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6; Xt = (X[tr] - mu) / sd; Xv = (X[~tr] - mu) / sd
        w = np.linalg.solve(Xt.T @ Xt + lam * np.eye(X.shape[1]), Xt.T @ (y[tr] - y[tr].mean())); p[~tr] = Xv @ w
    return p
pc = oof(z['n320'], np.log1p(ngt)); pc_gt = ngt + 1e-3 * np.random.RandomState(1).rand(I)
C = [M5, M, L]; cost = np.array([3.47, 5.05, 6.55])  # T4 real-input basis (agent 7)
F = lambda t: 0.5261 + 0.0102 * (t - 5.05)
def rep(tag, a):
    ap, sml = mix_ap(C, a); t = cost[a].mean() + 0.18; sh = np.bincount(a, minlength=3) / I
    print(f'{tag:58s} AP {ap:.4f} S/M/L {sml[0]:.4f}/{sml[1]:.4f}/{sml[2]:.4f} shares {np.round(sh,3)} T4 avg {t:.2f} AP-bar {ap-F(t)-0.003:+.4f}', flush=True); return ap
q, qL = 0.50, 0.399; n5, nL = int(q * I), int(round(qL * I))
for sig, nm in ((pc, 'n320'), (pc_gt, 'GT count')):
    o = np.argsort(sig); res = {}
    for r512 in (0, 1):
        for rL in (0, 1):
            vals = []
            for d in range(3 if (r512 or rL) else 1):
                rng = np.random.RandomState(200 + d); a = np.ones(I, int)
                lo = o[:n5] if not r512 else rng.permutation(I)[:n5]
                a[lo] = 0; rest = np.where(a == 1)[0]
                if not rL: hi = [i for i in o[::-1] if a[i] == 1][:nL]
                else: hi = rng.permutation(rest)[:nL]
                a[np.array(hi)] = 2
                vals.append(rep(f'{nm}: 512 {"random" if r512 else "count"}, L {"random" if rL else "count"} (draw {d})', a))
            res[(r512, rL)] = np.mean(vals)
    print(f'== {nm}: full exchange {res[(0,0)]:.4f}; resolution routed only {res[(0,1)]:.4f}; depth routed only {res[(1,0)]:.4f}; null {res[(1,1)]:.4f}')
    print(f'   lift over null: both {res[(0,0)]-res[(1,1)]:+.4f}; from the 512 decision {res[(0,1)]-res[(1,1)]:+.4f}; from the L decision {res[(1,0)]-res[(1,1)]:+.4f}; L decision given routed 512 {res[(0,0)]-res[(0,1)]:+.4f}')
def ap_subset(cache, mask):
    K = len(cache['catIds']); A = 4; sub = np.where(mask)[0]
    ev = [cache['evalImgs'][k * A * I + a * I + i] for k in range(K) for a in range(A) for i in sub]
    E = COCOeval(g, None, 'bbox'); E.params.imgIds = [cache['imgIds'][i] for i in sub]; E.params.catIds = cache['catIds']; E.evalImgs = ev; E._paramsEval = copy.deepcopy(E.params)
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0]
print('--- AP by GT-count bucket: M@512, M@640, loss (cf. depth gain flat at +0.014..+0.019, round 1 table 1b)')
for lo, hi in ((1, 3), (3, 6), (6, 11), (11, 10**6)):
    m = (ngt >= lo) & (ngt < hi); a, b = ap_subset(M5, m), ap_subset(M, m)
    print(f'  n_gt in [{lo},{hi}): {m.sum():4d} img  M512 {a:.4f}  M640 {b:.4f}  loss {b-a:+.4f}', flush=True)
