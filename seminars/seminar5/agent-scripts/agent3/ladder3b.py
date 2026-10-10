# Level ladder {no-P3, full, finer level}: the finer level is proxied by YOLO26-M at 768 (public weights).
# Images with the fewest small objects skip P3; the most small-heavy take the finer branch.
import numpy as np, sys
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent3'); from mix import score_multi, load, imgpos
W = '/data/tmp/ds-yolo/seminar5/work/agent3/'
g = np.load(W + 'gt_counts.npz'); ids = g['ids']; C = g['C']; n = len(ids); R = np.load(W + 'router_scores.npz')
small = (C[:, 0] + C[:, 1]).astype(float); order = np.array([imgpos[int(i)] for i in ids])
V = [load('m', 'full'), load('m', sys.argv[1]), load('m768', 'full')]
print('768 alone', round(score_multi(V, np.full(n, 2))[0], 4), flush=True)
F = lambda L: 0.5261 + 0.0102 * (L - 5.36); tb = np.random.default_rng(1).random(n) * 1e-6
def assign(key, s_skip, s_up):
    o = np.argsort(key + tb); a = np.zeros(n, int); a[o[:int(round(s_skip * n))]] = 1
    if s_up > 0: a[o[n - int(round(s_up * n)):]] = 2
    out = np.zeros(n, int); out[order] = a; return out
for s_skip, s_up in [(0.0, 0.2), (0.49, 0.2), (0.40, 0.25), (0.30, 0.20)]:
    res = []
    for nm, k in [('GT', small), ('m640', R['m640_log1p_small'])]:
        res.append(f'{nm} {score_multi(V, assign(k, s_skip, s_up))[0]:.4f}')
    nl = np.mean([score_multi(V, assign(np.random.default_rng(300 + r).random(n), s_skip, s_up))[0] for r in range(2)])
    bars = ' / '.join(f'{F(5.36 - 1.0 * s_skip + c * s_up) + 0.003:.4f}' for c in (1.3, 1.6))
    print(f'{sys.argv[1]} skip {s_skip:.2f} up {s_up:.2f}: ' + ' | '.join(res) + f' | null {nl:.4f} | bar at up-cost 1.3 / 1.6 ms: {bars}', flush=True)
