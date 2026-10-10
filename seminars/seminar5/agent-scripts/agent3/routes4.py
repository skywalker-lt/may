# Two-bit level mask {P3 on/off} x {P5 on/off}: GT rule, learned (m640 / n320 OOF ridge) and share-matched null.
# Branch costs (T4 est., unset basis, from the per-layer attribution): skip P3 saves d3, skip P5 saves d5.
import numpy as np, sys
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent3'); from mix import score_multi, load, imgpos
W = '/data/tmp/ds-yolo/seminar5/work/agent3/'
g = np.load(W + 'gt_counts.npz'); ids = g['ids']; C = g['C']; n = len(ids); R = np.load(W + 'router_scores.npz')
small = C[:, 0] + C[:, 1]; large = C[:, 3]
order = np.array([imgpos[int(i)] for i in ids])  # npz order -> evalImgs order
t3, t5 = sys.argv[1], sys.argv[2]   # e.g. noP3_32 noP5_128 ; P4only variant name
V = [load('m', 'full'), load('m', t3), load('m', t5), load('m', sys.argv[3])]
d3, d5 = 1.00, 0.52
F = lambda L: 0.5261 + 0.0102 * (L - 5.36)
rng = np.random.default_rng(1); tb = rng.random(n) * 1e-6
def assign_from(k3, k5, s3, s5):
    b3 = np.zeros(n, bool); b3[np.argsort(k3 + tb)[:int(round(s3 * n))]] = True
    b5 = np.zeros(n, bool); b5[np.argsort(k5 + tb)[:int(round(s5 * n))]] = True
    a = np.where(b3 & b5, 3, np.where(b3, 1, np.where(b5, 2, 0)))
    out = np.zeros(n, int); out[order] = a; return out, b3.mean(), b5.mean()
for s3, s5 in [(0.49, 0.0), (0.49, 0.15), (0.49, 0.27), (0.40, 0.20), (0.60, 0.27)]:
    L = 5.36 - d3 * s3 - d5 * s5; bar = F(L) + 0.003
    row = []
    for name, k3, k5 in [('GT', small.astype(float), large.astype(float)), ('m640', R['m640_log1p_small'], R['m640_log1p_large']),
                         ('n320', R['n320_log1p_small'], R['n320_log1p_large'])]:
        a, _, _ = assign_from(k3, k5, s3, s5); row.append(f'{name} {score_multi(V, a)[0]:.4f}')
    nl = []
    for r in range(2):
        rr = np.random.default_rng(200 + r); a, _, _ = assign_from(rr.random(n), rr.random(n), s3, s5); nl.append(score_multi(V, a)[0])
    print(f'{t3}+{t5} s3 {s3:.2f} s5 {s5:.2f}: avg {L:.3f} ms bar {bar:.4f} | ' + ' | '.join(row) + f' | null {np.mean(nl):.4f}', flush=True)
