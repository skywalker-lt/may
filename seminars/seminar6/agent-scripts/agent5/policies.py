"""Lens 5 headroom: rect (native-aspect) policies on one YOLO26-L / M weight set, scored exactly from the square dumps
(a rect engine at long side s sees the same content as the square letterbox at s; only the grey padding differs) and
costed with the est. T4 pixel model.  Output: policies.log."""
import numpy as np, time, sys
from mix import *

def section(t): print('\n## ' + t, flush=True)
t0 = time.time()
RUNGS_L = [448, 512, 544, 576, 640]; RUNGS_M = [448, 512, 576, 608, 640]

section('A. Image shapes and rect pixel saving (val2017)')
frac = np.minimum(W, H) / np.maximum(W, H)
print(f'images {I}; mean short/long {frac.mean():.3f}; landscape {int((W>H).sum())} portrait {int((W<H).sum())} square {int((W==H).sum())}')
for s in [640, 544, 512]:
    pix = np.array([np.prod(rect_shape(s, W[n], H[n])) for n in range(I)]) / 1000
    print(f'  long side {s}: mean rect kpix {pix.mean():.1f} (square {s*s/1000:.1f}; ratio {pix.mean()/(s*s/1000):.3f}); max {pix.max():.1f}')

section('B. Fixed-long-side rect points (AP = square dump AP exactly; cost est. by the pixel model, interp / linfit)')
rect_pts = {'l': [], 'm': []}
HAVE_M = all(os.path.exists(f'{CACHE}/m{s}.pkl') for s in RUNGS_M)
for model, rungs in (('l', RUNGS_L),) + ((('m', RUNGS_M),) if HAVE_M else ()):
    for s in rungs:
        ap = dump_stats(f'{model}{s}')[0]
        mi, mx = policy_cost(model, np.full(I, s), cost_interp); li, lx = policy_cost(model, np.full(I, s), cost_linfit)
        sq = min(SQ_PTS[model], key=lambda p: abs(p[1] - ap))[0] if s != 640 or model == "m" else 6.89
        env_sq = env_at(ALL_SQ, mi)
        print(f'  {model.upper()} rect long {s}: AP {ap:.4f}; square T4 {sq:.2f} meas; rect est. avg {mi:.2f} (linfit {li:.2f}), worst {mx:.2f}; '
              f'square envelope at {mi:.2f} ms = {env_sq:.4f}; margin vs env+0.003 {ap - env_sq - 0.003:+.4f}')
        rect_pts[model].append((mi, ap, f'{model}{s} rect'))

section('C. Pixel-cap rect policies on L (dense: long side = largest rung whose rect shape fits the cap; no router)')
cap_pts = []
for cap in [200, 230, 262, 296, 332, 410]:
    scales = np.zeros(I, int)
    for n in range(I):
        ok = [s for s in RUNGS_L if np.prod(rect_shape(s, W[n], H[n])) / 1000 <= cap + 1e-9]
        scales[n] = max(ok) if ok else 448
    names = np.array([f'l{s}' for s in scales]); st = mix_ap(list(names))
    mi, mx = policy_cost('l', scales, cost_interp); li, lx = policy_cost('l', scales, cost_linfit)
    shares = {s: round(float((scales == s).mean()), 2) for s in RUNGS_L if (scales == s).any()}
    env_sq = env_at(ALL_SQ, mi)
    print(f'  cap {cap} kpix: shares {shares}; {fmt(st)}; est. avg {mi:.2f} (linfit {li:.2f}) worst {mx:.2f}; sq-env {env_sq:.4f}; '
          f'margin vs sq-env+0.003 {st[0]-env_sq-0.003:+.4f}  [{time.time()-t0:.0f}s]', flush=True)
    cap_pts.append((mi, st[0], f'cap{cap}'))

RECT_ENV = [(p[0], p[1]) for p in rect_pts['l'] + rect_pts['m']] + [(p[0], p[1]) for p in cap_pts]
section('D. The rect envelope (hull over fixed-long-side and pixel-cap points, est. costs) against the square envelope')
for t in [4.0, 4.3, 4.6, 4.9, 5.3, 5.6, 6.0, 6.5]:
    print(f'  t={t:.1f} ms: square env {env_at(ALL_SQ, t):.4f}; rect env {env_at(RECT_ENV, t):.4f}; diff {env_at(RECT_ENV, t)-env_at(ALL_SQ, t):+.4f}')
print('  rect hull points:', [(round(x, 2), round(y, 4)) for x, y in hull(RECT_ENV)])

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

section('F. M for the low budget: M rect fixed points')
for s in (RUNGS_M if HAVE_M else []):
    ap = dump_stats(f'm{s}')[0]; mi, mx = policy_cost('m', np.full(I, s), cost_interp)
    print(f'  M rect long {s}: AP {ap:.4f}; est. avg {mi:.2f} worst {mx:.2f}; sq-env {env_at(ALL_SQ, mi):.4f}; rect-env {env_at(RECT_ENV, mi):.4f}')
print(f'done {time.time()-t0:.0f}s')
