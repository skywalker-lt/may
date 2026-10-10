"""Section E of policies.py, rerun alone (the first run was killed with its background task)."""
import numpy as np, time
from mix import *
def section(t): print('\n## ' + t, flush=True)
t0 = time.time()
RECT_ENV = [(3.33, 0.4937), (3.36, 0.5063), (3.71, 0.5203), (4.04, 0.5234), (4.39, 0.5261), (4.28, 0.5117), (4.32, 0.5262),
        (4.51, 0.5329), (4.79, 0.5368), (5.48, 0.5417), (4.28, 0.5240), (4.47, 0.5310), (4.70, 0.5364), (5.01, 0.5401), (5.42, 0.5418)]
section('E. Routed rect ladders on L: count router picks the long side per image (engine table on the host, no If; router 0.18 ms charged)')
# out-of-fold ridge router on the pooled N@320 stem feature, target log1p(S+M count); 5 folds
z = np.load(f'{IN}/dumps/val2017_stem_pooled.npz'); zi = {int(i): n for n, i in enumerate(z['image_id'])}
X = np.stack([z['n320'][zi[i]] for i in imgIds]).astype(np.float64)
X = (X - X.mean(0)) / (X.std(0) + 1e-6); Xb = np.hstack([X, np.ones((I, 1))])
y = np.log1p(cnt_sm); rng = np.random.RandomState(0); folds = rng.permutation(I) % 5
pred = np.zeros(I)
for f in range(5):
    tr = folds != f; te = ~tr
    lam = 30.0; A_ = Xb[tr].T @ Xb[tr] + lam * np.eye(Xb.shape[1]); wv = np.linalg.solve(A_, Xb[tr].T @ y[tr]); pred[te] = Xb[te] @ wv
from scipy.stats import spearmanr
print(f'  OOF ridge router: Spearman with GT S+M count {spearmanr(pred, cnt_sm)[0]:.3f}; GT rule Spearman 1.0')
rng2 = np.random.RandomState(1)
def ladder(lo, hi, q, signal, label, rounds=1):
    """send the predicted-sparse share q to the lo rung."""
    thr = np.quantile(signal, q); scales = np.where(signal <= thr, lo, hi)
    st = mix_ap([f'l{s}' for s in scales]); mi, mx = policy_cost('l', scales, cost_interp); li, _ = policy_cost('l', scales, cost_linfit)
    mi += ROUTER_MS; li += ROUTER_MS; mx += ROUTER_MS
    # share-matched null: random route at the same share (mean of `rounds` draws)
    nulls = []
    for r in range(rounds):
        perm = rng2.permutation(I); sc_n = np.full(I, hi); sc_n[perm[:int(round(q * I))]] = lo
        nulls.append(mix_ap([f'l{s}' for s in sc_n])[0])
    null = float(np.mean(nulls))
    e_sq = env_at(ALL_SQ, mi); e_rect = env_at(RECT_ENV, mi)
    print(f'  L {lo}/{hi} q={q:.2f} [{label}]: {fmt(st)}; est. avg {mi:.2f} (linfit {li:.2f}) worst {mx:.2f}; null {null:.4f}; '
          f'vs null+0.003 {st[0]-null-0.003:+.4f}; sq-env {e_sq:.4f} margin {st[0]-e_sq-0.003:+.4f}; rect-env {e_rect:.4f} margin {st[0]-e_rect-0.003:+.4f}  [{time.time()-t0:.0f}s]', flush=True)
    return st[0], mi
for lo, hi, q in [(512, 640, 0.4), (512, 640, 0.6), (448, 640, 0.6), (544, 640, 0.6)]:
    ladder(lo, hi, q, cnt_sm + 1e-3 * rng2.rand(I), 'GT S+M count')
    ladder(lo, hi, q, pred, 'OOF ridge n320')

print(f'done {time.time()-t0:.0f}s')
