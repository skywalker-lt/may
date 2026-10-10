# Round 2: same-footing comparison of cheap rungs (B = no-P3 proxy at 640, M@512, M@512 without P3) and expensive
# rungs (C proxy = M@768, L@640), exact per-image mixtures of evalImgs. T4 unset basis (dense M 5.36 ms).
import numpy as np, sys
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent3'); from mix import score_multi, load, imgpos
W = '/data/tmp/ds-yolo/seminar5/work/agent3/'; D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'
g = np.load(W + 'gt_counts.npz'); ids = g['ids']; C = g['C'].astype(float); n = len(ids)
R = np.load(W + 'router_scores.npz'); z = np.load(D + 'val2017_stem_pooled.npz')
zid = z['image_id']
order = np.array([imgpos[int(i)] for i in ids])
fold = np.random.default_rng(0).permutation(n) % 5
def ridge_oof(X, y, lam=10.0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = np.hstack([X, np.ones((n, 1))]); p = np.zeros(n)
    for f in range(5):
        tr = fold != f; A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1]); A[-1, -1] -= lam
        p[~tr] = X[~tr] @ np.linalg.solve(A, X[tr].T @ y[tr])
    return p
key = {}
key['m640_small'] = R['m640_log1p_small']; key['n320_small'] = R['n320_log1p_small']
# features must be in the same image order as gt_counts ids
if zid is not None:
    pos = {int(i): j for j, i in enumerate(zid)}; sel = np.array([pos[int(i)] for i in ids])
    Xn, Xm = z['n320'][sel].astype(float), z['m640'][sel].astype(float)
    tgt = np.log1p(C[:, 0] + C[:, 1] + C[:, 2])            # non-large count (agent 1's shrink target)
    key['n320_nonlarge'] = ridge_oof(Xn, tgt); key['m640_nonlarge'] = ridge_oof(Xm, tgt)
    key['n320_all'] = ridge_oof(Xn, np.log1p(C.sum(1)))
key['GT_small'] = C[:, 0] + C[:, 1]; key['GT_nonlarge'] = C[:, 0] + C[:, 1] + C[:, 2]
tb = np.random.default_rng(1).random(n) * 1e-6
V = {'A': load('m', 'full'), 'B16': load('m', 'noP3_16'), 'B32': load('m', 'noP3_32'), 'C768': load('m768', 'full'),
     'S512': load('m512', 'full'), 'S512b16': load('m512', 'noP3_16'), 'S512b32': load('m512', 'noP3_32'),
     'L': load('l', 'full')}
names = list(V); VL = [V[k] for k in names]
def route(k, rungs):
    """rungs: list of (variant, share) from the lowest predicted count upward; remaining images take A."""
    o = np.argsort(k + tb); a = np.full(n, names.index('A')); p = 0
    for v, s in rungs:
        m = int(round(s * n)); a[o[p:p + m]] = names.index(v); p += m
    out = np.zeros(n, int); out[order] = a; return out
def route_top(k, low, high):
    o = np.argsort(k + tb); a = np.full(n, names.index('A')); p = 0
    for v, s in low:
        m = int(round(s * n)); a[o[p:p + m]] = names.index(v); p += m
    q = n
    for v, s in high:
        m = int(round(s * n)); a[o[q - m:q]] = names.index(v); q -= m
    out = np.zeros(n, int); out[order] = a; return out
def sc(assign): return score_multi(VL, assign)
def null(low, high, r=3):
    return np.mean([sc(route_top(np.random.default_rng(500 + i).random(n), low, high))[0] for i in range(r)])

def show(tag, low, high, kk, do_null=False):
    r = sc(route_top(key[kk], low, high))
    nl = f' | null {null(low, high, 1):.4f}' if do_null else ''
    print(f'{tag:10s} low {low} high {high} router {kk}: AP {r[0]:.4f} S/M/L {r[1]:.4f}/{r[2]:.4f}/{r[3]:.4f}{nl}', flush=True)
for s in (0.5, 0.4):
    show('cheap', [('B16', s)], [], 'm640_small'); show('cheap', [('B32', s)], [], 'm640_small')
    show('cheap', [('S512', s)], [], 'n320_nonlarge'); show('cheap', [('S512b16', s)], [], 'n320_nonlarge', s == 0.5)
show('cheap', [('S512', 0.5)], [], 'm640_nonlarge'); show('cheap', [('S512', 0.5)], [], 'GT_nonlarge')
show('cheap', [('S512b32', 0.5)], [], 'n320_nonlarge')
show('stack', [('S512b16', 0.2), ('S512', 0.3)], [], 'n320_nonlarge', True)
show('stack', [('S512b16', 0.3), ('S512', 0.3)], [], 'n320_nonlarge')
show('stack', [('S512b32', 0.2), ('S512', 0.3)], [], 'n320_nonlarge')
show('ladder', [('S512', 0.5)], [('L', 0.4)], 'n320_all', True)
show('ladder', [('B16', 0.49)], [('L', 0.30)], 'm640_small', True)
show('ladder', [('B32', 0.49)], [('L', 0.30)], 'm640_small')
show('ladder', [('B16', 0.49)], [('C768', 0.20)], 'm640_small')
show('ladder4', [('S512b16', 0.2), ('S512', 0.3)], [('L', 0.45)], 'n320_all', True)
show('ladder4', [('S512b16', 0.2), ('S512', 0.3)], [('L', 0.45)], 'n320_nonlarge')
print('CDONE', flush=True)
