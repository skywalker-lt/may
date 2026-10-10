# reproduce seminar-4 two-scale rows, then explore three-scale routes at equal average T4 latency
import numpy as np, mixlib as M, feats as F
from scipy.stats import spearmanr
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dumpml_yolo26m_768_coco")]
ids, ns, nm, nl, minA = F.gt_counts(); assert list(ids) == srcs[0]["imgIds"]
Xn, Xm = F.features(ids); I = len(ids)
for lam in (1, 10, 100):
    p = F.oof_ridge(Xn, np.log1p(ns), lam); print("n320 ridge lam", lam, "spearman log1p(n_small)", round(spearmanr(p, ns).correlation, 3))
pn = F.oof_ridge(Xn, np.log1p(ns), 10); pm = F.oof_ridge(Xm, np.log1p(ns), 10)
print("m640 spearman", round(spearmanr(pm, ns).correlation, 3))
def top(score, share, rng):  # top share by score (ties broken randomly)
    o = np.lexsort((rng.rand(I), -score)); sel = np.zeros(I, bool); sel[o[:int(round(share * I))]] = True; return sel
rng = np.random.RandomState(1)
for share in (0.43, 0.52):
    gtr = top(np.log1p(ns) + 1e-3 * nm, share, rng); lr = top(pn, share, rng)
    print(f"512/768 share {share}: GT small rule", M.score(srcs, np.where(gtr, 2, 0)), " learned n320", M.score(srcs, np.where(lr, 2, 0)))
    nulls = [M.score(srcs, np.where(rng.rand(I) < 1, np.isin(np.arange(I), rng.permutation(I)[:int(round(share*I))]) * 2, 0))[0] for _ in range(5)]
    print("   null draws", nulls, "mean", round(np.mean(nulls), 4), "sd", round(np.std(nulls, ddof=1), 4))
