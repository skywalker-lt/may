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
mode = sys.argv[1]
if mode == 'alone':
    for v in names: print(v, 'alone', [round(x, 4) for x in sc(np.full(n, names.index(v)))], flush=True)
if mode == 'cheap':
    for s in (0.3, 0.4, 0.5, 0.6):
        for v, kk in [('B16', 'm640_small'), ('B32', 'm640_small'), ('S512', 'n320_nonlarge'), ('S512', 'n320_small'),
                      ('S512', 'm640_nonlarge'), ('S512b16', 'n320_nonlarge'), ('S512b32', 'n320_nonlarge'),
                      ('B16', 'GT_small'), ('B32', 'GT_small'), ('S512', 'GT_nonlarge')]:
            r = sc(route(key[kk], [(v, s)]))
            print(f'cheap share {s:.2f} {v:8s} router {kk:14s} AP {r[0]:.4f} S/M/L {r[1]:.4f}/{r[2]:.4f}/{r[3]:.4f}', flush=True)
        for v in ('B16', 'B32', 'S512', 'S512b16'):
            print(f'cheap share {s:.2f} {v:8s} null AP {null([(v, s)], []):.4f}', flush=True)
if mode == 'stack':   # 512-without-P3 for the sparsest, 512 next, 640 rest (one key)
    for q1, q2 in [(0.1, 0.4), (0.2, 0.3), (0.2, 0.4), (0.3, 0.3), (0.15, 0.35)]:
        for v in ('S512b16', 'S512b32'):
            r = sc(route(key['n320_nonlarge'], [(v, q1), ('S512', q2)]))
            print(f'stack {v} {q1:.2f} + S512 {q2:.2f}: AP {r[0]:.4f} S/M/L {r[1]:.4f}/{r[2]:.4f}/{r[3]:.4f} | null {null([(v, q1), ("S512", q2)], []):.4f}', flush=True)
if mode == 'ladder':
    cfgs = [('B16', 'm640_small', 0.49, 'C768', 0.20), ('B32', 'm640_small', 0.49, 'C768', 0.20),
            ('B16', 'm640_small', 0.49, 'L', 0.30), ('B32', 'm640_small', 0.49, 'L', 0.30),
            ('S512', 'n320_all', 0.50, 'L', 0.40), ('S512', 'n320_all', 0.40, 'L', 0.30),
            ('S512', 'n320_nonlarge', 0.50, 'L', 0.40), ('S512', 'n320_nonlarge', 0.50, 'C768', 0.30)]
    for lo, kk, s1, hi, s2 in cfgs:
        r = sc(route_top(key[kk], [(lo, s1)], [(hi, s2)]))
        print(f'ladder {lo} {s1:.2f} / A / {hi} {s2:.2f} router {kk}: AP {r[0]:.4f} S/M/L {r[1]:.4f}/{r[2]:.4f}/{r[3]:.4f} | null {null([(lo, s1)], [(hi, s2)]):.4f}', flush=True)
    for (a1, s1), (a2, s2), s3 in [(('S512b16', 0.2), ('S512', 0.3), 0.45), (('S512b16', 0.2), ('S512', 0.35), 0.45),
                                   (('S512b32', 0.2), ('S512', 0.3), 0.45), (('S512b16', 0.15), ('S512', 0.35), 0.45)]:
        r = sc(route_top(key['n320_all'], [(a1, s1), (a2, s2)], [('L', s3)]))
        print(f'ladder4 {a1} {s1:.2f} + S512 {s2:.2f} / A / L {s3:.2f} router n320_all: AP {r[0]:.4f} S/M/L {r[1]:.4f}/{r[2]:.4f}/{r[3]:.4f} | null {null([(a1, s1), (a2, s2)], [("L", s3)]):.4f}', flush=True)
