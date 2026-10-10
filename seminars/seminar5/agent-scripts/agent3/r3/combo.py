# L@512 with routed P3 drop inside the 512 rung, L@640 for the rest: scale by n320 non-large (pre-backbone),
# B by m640 small count among the 512 images (post-backbone, free). T4 unset est.; B saves 0.179 x L@512 time.
import numpy as np, sys
sys.path.insert(0, '.'); from mix3 import key, n, order, score_multi, load, tb
from envelope2 import env, F, tL
V = [load('l512', 'noP3_16'), load('l512', 'noP3_32'), load('l512', 'full'), load('l', 'full')]
for q in [0.5, 0.6, 0.7]:
    o = np.argsort(key['n320_nonlarge'] + tb); low = o[:int(round(q * n))]
    for fb in [0.2, 0.3]:
        lo2 = low[np.argsort(key['m640_small'][low] + tb[low])][:int(round(fb * n))]
        for bi, nm in [(0, 'tau16'), (1, 'tau32')]:
            a = np.full(n, 3); a[low] = 2; a[lo2] = bi; out = np.zeros(n, int); out[order] = a
            ap = score_multi(V, out)[0]
            t0 = q * tL(512) + (1 - q) * 6.89 - fb * 0.179 * tL(512)
            cells = [f'r {r}: {t0 + r:.2f} ms bar {ap - F(t0 + r) - 0.003:+.4f} env+0.003 {ap - env(t0 + r) - 0.003:+.4f}' for r in (0.064, 0.178)]
            print(f'q512 {q} B {fb} {nm}: AP {ap:.4f} | ' + ' | '.join(cells), flush=True)
