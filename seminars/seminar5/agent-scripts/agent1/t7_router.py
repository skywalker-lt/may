# router target variants for the shrink decision ({512,640}, shares 0.5 and 0.6), n320 thumbnail feature, out-of-fold
import numpy as np, mixlib as M, feats as F
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dumpml_yolo26m_768_coco")]
ids, ns, nm, nl, minA = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
rng = np.random.RandomState(3)
def shrink(score, s5):  # lowest score -> 512
    ch = np.ones(I, int); o = np.lexsort((rng.rand(I), score)); ch[o[:int(round(s5 * I))]] = 0; return ch
targets = {"log1p(S+M) lam10": (np.log1p(ns + nm), 10), "log1p(S+M) lam1": (np.log1p(ns + nm), 1), "log1p(S+M) lam100": (np.log1p(ns + nm), 100),
           "1[S+M>0] lam10": ((ns + nm > 0).astype(float), 10), "log1p(S)+0.5log1p(M) lam10": (np.log1p(ns) + 0.5 * np.log1p(nm), 10),
           "log1p(S) lam10": (np.log1p(ns), 10), "-log(minArea) lam10": (-np.log(np.minimum(minA, 640 * 640)), 10)}
for name, (y, lam) in targets.items():
    p = F.oof_ridge(Xn, y, lam); r = [M.score(srcs, shrink(p, s))[0] for s in (0.5, 0.6)]
    print(f"{name:32s} share0.5 {r[0]:.4f} share0.6 {r[1]:.4f}", flush=True)
for name, sc in (("GT log1p(S+M)", np.log1p(ns + nm) + 1e-3 * nl), ("m640 ridge log1p(S+M) (not in engine)", F.oof_ridge(Xm, np.log1p(ns + nm), 10))):
    r = [M.score(srcs, shrink(sc, s))[0] for s in (0.5, 0.6)]; print(f"{name:32s} share0.5 {r[0]:.4f} share0.6 {r[1]:.4f}", flush=True)
