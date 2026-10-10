"""ISO-K: iso-pixel shape route (exact aspect, free) plus a count bit choosing between two pixel caps.
Comparators: the dense rect envelope (fixed long side + dense caps, from policies.log), the dense cap chord at equal
average cost, and a random bit at the same share.  Exact mixing; est. T4 rect costs; router 0.18 ms charged."""
import numpy as np, time
from mix import *
t0 = time.time()
RUNGS = [448, 512, 544, 576, 640]
def cap_scales(cap):
    out = np.zeros(I, int)
    for n in range(I):
        ok = [s for s in RUNGS if np.prod(rect_shape(s, W[n], H[n])) / 1000 <= cap + 1e-9]
        out[n] = max(ok) if ok else 448
    return out
# dense cap points from policies.log (AP exact, cost est.), plus the fixed-long-side rect points: the rect envelope
RECT = [(3.33, 0.4937), (3.36, 0.5063), (3.71, 0.5203), (4.04, 0.5234), (4.39, 0.5261), (4.28, 0.5117), (4.32, 0.5262),
        (4.51, 0.5329), (4.79, 0.5368), (5.48, 0.5417), (4.28, 0.5240), (4.47, 0.5310), (4.70, 0.5364), (5.01, 0.5401), (5.42, 0.5418)]
z = np.load(f'{IN}/dumps/val2017_stem_pooled.npz'); zi = {int(i): n for n, i in enumerate(z['image_id'])}
X = np.stack([z['n320'][zi[i]] for i in imgIds]).astype(np.float64); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
Xb = np.hstack([X, np.ones((I, 1))]); y = np.log1p(cnt_sm); folds = np.random.RandomState(0).permutation(I) % 5; pred = np.zeros(I)
for f in range(5):
    tr = folds != f; A_ = Xb[tr].T @ Xb[tr] + 30.0 * np.eye(Xb.shape[1]); pred[~tr] = Xb[~tr] @ np.linalg.solve(A_, Xb[tr].T @ y[tr])
rng = np.random.RandomState(3)
for lo_cap, hi_cap, q in [(230, 332, 0.5), (200, 332, 0.5), (230, 410, 0.6)]:
    s_lo, s_hi = cap_scales(lo_cap), cap_scales(hi_cap)
    for label, sig in [('GT S+M count', cnt_sm + 1e-3 * rng.rand(I)), ('OOF ridge n320', pred)]:
        bit = sig > np.quantile(sig, q); sc = np.where(bit, s_hi, s_lo)
        st = mix_ap([f'l{s}' for s in sc]); c, mx = policy_cost('l', sc); c += ROUTER_MS; mx += ROUTER_MS
        pr = rng.permutation(I); rbit = np.zeros(I, bool); rbit[pr[:bit.sum()]] = True
        nul = mix_ap([f'l{s}' for s in np.where(rbit, s_hi, s_lo)])[0]
        e = env_at(RECT, c)
        print(f'ISO-K caps {lo_cap}/{hi_cap} q={q} [{label}]: {fmt(st)}; est. avg {c:.2f} worst {mx:.2f}; random-bit null {nul:.4f} '
              f'(route - null {st[0]-nul:+.4f}); rect env {e:.4f} (margin vs env+0.003 {st[0]-e-0.003:+.4f}); sq env {env_at(ALL_SQ, c):.4f}  [{time.time()-t0:.0f}s]', flush=True)
print(f'done {time.time()-t0:.0f}s')
