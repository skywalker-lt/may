# Round 3 mixtures (exact evalImgs recombination). usage: mix3.py <job>
#  lvl:<tag>        base <tag>_full, alt <tag>_noP3_16 / _noP3_32; low predicted small count -> alt
#  rs2l             L@512 for low predicted non-large count, L@640 for the rest (and L@576 middle rung)
#  dump:<tag>:<base> alt = <tag>_full (a served dump), base = <base>_full
import numpy as np, sys
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent3'); from mix import score_multi, load, imgpos
W = '/data/tmp/ds-yolo/seminar5/work/agent3/'; D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'
g = np.load(W + 'gt_counts.npz'); ids = g['ids']; C = g['C'].astype(float); n = len(ids)
R = np.load(W + 'router_scores.npz'); z = np.load(D + 'val2017_stem_pooled.npz')
order = np.array([imgpos[int(i)] for i in ids]); fold = np.random.default_rng(0).permutation(n) % 5
def ridge_oof(X, y, lam=10.0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = np.hstack([X, np.ones((n, 1))]); p = np.zeros(n)
    for f in range(5):
        tr = fold != f; A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1]); A[-1, -1] -= lam
        p[~tr] = X[~tr] @ np.linalg.solve(A, X[tr].T @ y[tr])
    return p
pos = {int(i): j for j, i in enumerate(z['image_id'])}; sel = np.array([pos[int(i)] for i in ids])
Xn = z['n320'][sel].astype(float)
key = {'m640_small': R['m640_log1p_small'], 'n320_small': R['n320_log1p_small'], 'GT_small': C[:, 0] + C[:, 1],
       'n320_nonlarge': ridge_oof(Xn, np.log1p(C[:, 0] + C[:, 1] + C[:, 2])), 'GT_nonlarge': C[:, 0] + C[:, 1] + C[:, 2]}
tb = np.random.default_rng(1).random(n) * 1e-6
def assign(k, rungs):  # rungs from the lowest key upward: list of (index, share); remainder -> index of last
    o = np.argsort(k + tb); a = np.full(n, rungs[-1][0]); p = 0
    for v, s in rungs[:-1]:
        m = int(round(s * n)); a[o[p:p + m]] = v; p += m
    out = np.zeros(n, int); out[order] = a; return out
def run(V, rungs, keys, nulls=3, label=''):
    out = []
    for kn in keys:
        r = score_multi(V, assign(key[kn], rungs)); out.append(f'{kn} {r[0]:.4f} (S/M/L {r[1]:.4f}/{r[2]:.4f}/{r[3]:.4f})')
    nl = [score_multi(V, assign(np.random.default_rng(700 + i).random(n), rungs))[0] for i in range(nulls)]
    print(f'{label} rungs {rungs}: ' + ' | '.join(out) + f' | null {np.mean(nl):.4f} (sd {np.std(nl):.4f})', flush=True)
job = sys.argv[1] if __name__ == '__main__' else ''
if job.startswith('lvl:'):
    t = job[4:]; base = load(t, 'full')
    for v in ['noP3_16', 'noP3_32']:
        V = [load(t, v), base]
        for s in [0.3, 0.4, 0.49, 0.6]: run(V, [(0, s), (1, 1 - s)], ['GT_small', 'm640_small', 'n320_small'], label=f'{t} {v}')
elif job == 'rs2l':
    V = [load('l512', 'full'), load('l576', 'full'), load('l', 'full')]
    for s in [0.3, 0.4, 0.5, 0.6, 0.7]: run(V, [(0, s), (2, 1 - s)], ['GT_nonlarge', 'n320_nonlarge'], label='RS-2L 512/640')
    for s in [0.3, 0.5, 0.7]: run(V, [(1, s), (2, 1 - s)], ['GT_nonlarge', 'n320_nonlarge'], label='L 576/640')
    for s in [0.3, 0.5]: run(V, [(0, s), (1, 1 - s)], ['GT_nonlarge', 'n320_nonlarge'], label='L 512/576')
    for s5, s6 in [(0.3, 0.4), (0.4, 0.3), (0.5, 0.3)]: run(V, [(0, s5), (1, s6), (2, 1 - s5 - s6)], ['GT_nonlarge', 'n320_nonlarge'], label='L 512/576/640')
elif job.startswith('dump:'):
    _, t, b = job.split(':'); V = [load(t, 'full'), load(b, 'full')]
    for s in [0.3, 0.49, 0.6]: run(V, [(0, s), (1, 1 - s)], ['GT_small', 'm640_small'], label=f'{t} vs {b}')
if job: print('MIXDONE', job, flush=True)
