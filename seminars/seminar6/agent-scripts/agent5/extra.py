"""Free-signal routes (no router cost): native long side <= 500 px -> lower rung (upsampling adds no information),
and the iso-pixel cap scored against an equal-COST random null.  Exact mixing; est. T4 rect costs."""
import numpy as np, time
from mix import *
t0 = time.time()
ls = np.maximum(W, H); small = ls <= 500
print(f'native long side <= 500: {small.sum()} images ({small.mean():.3f}); GT objects per image there {cnt_all[small].mean():.2f} vs {cnt_all[~small].mean():.2f} elsewhere')
rng = np.random.RandomState(7)
for lo in (512, 544):
    sc = np.where(small, lo, 640); st = mix_ap([f'l{s}' for s in sc]); c, mx = policy_cost('l', sc)
    nul = []
    for r in range(2):
        p = rng.permutation(I); sn = np.full(I, 640); sn[p[:small.sum()]] = lo; nul.append(mix_ap([f'l{s}' for s in sn])[0])
    print(f'native-size route L {lo}/640: {fmt(st)}; est. avg {c:.2f} worst {mx:.2f}; random same-share null {np.mean(nul):.4f} '
          f'(draws {nul[0]:.4f}, {nul[1]:.4f}); route - null {st[0]-np.mean(nul):+.4f}; sq-env {env_at(ALL_SQ, c):.4f}  [{time.time()-t0:.0f}s]', flush=True)
print(f'done {time.time()-t0:.0f}s')
