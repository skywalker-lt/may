# three-scale {512,640,768} routes, one weight set (public M), at T4 average latency budgets; T4 real branch ms 3.47/5.05/7.12
import numpy as np, mixlib as M, feats as F
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dumpml_yolo26m_768_coco")]
ids, ns, nm, nl, minA = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
T = np.array([3.47, 5.05, 7.12])
pn_s = F.oof_ridge(Xn, np.log1p(ns), 10)                 # predicted small count
pn_sm = F.oof_ridge(Xn, np.log1p(ns + nm), 10)           # predicted small+medium count
rng = np.random.RandomState(7)
def route(score_hi, score_lo, s5, s7):
    """768 to the top-s7 of score_hi; 512 to the bottom-s5 of score_lo among the rest; 640 otherwise"""
    ch = np.ones(I, int); o = np.lexsort((rng.rand(I), -score_hi)); n7 = int(round(s7 * I)); ch[o[:n7]] = 2
    rest = np.where(ch == 1)[0]; o2 = rest[np.lexsort((rng.rand(len(rest)), score_lo[rest]))]; ch[o2[:int(round(s5 * I))]] = 0; return ch
def null(s5, s7, k=3):
    v = []
    for _ in range(k):
        p = rng.permutation(I); ch = np.ones(I, int); ch[p[:int(round(s7*I))]] = 2; ch[p[int(round(s7*I)):int(round(s7*I))+int(round(s5*I))]] = 0; v.append(M.score(srcs, ch)[0])
    return round(np.mean(v), 4), round(np.std(v, ddof=1), 4)
gt_hi = np.log1p(ns) + 1e-3 * nm; gt_lo = np.log1p(ns + nm) + 1e-3 * nl
print("budget = dense M real 5.05 ms; s7 = s5*(5.05-3.47)/(7.12-5.05)")
for s5 in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    s7 = s5 * (5.05 - 3.47) / (7.12 - 5.05)
    lat = s5 * T[0] + (1 - s5 - s7) * T[1] + s7 * T[2]
    r_gt = M.score(srcs, route(gt_hi, gt_lo, s5, s7)); r_ln = M.score(srcs, route(pn_s, pn_sm, s5, s7)); r_l1 = M.score(srcs, route(pn_s, pn_s, s5, s7))
    nl_ = null(s5, s7) if s5 > 0 else ("-",)
    print(f"s512 {s5:.2f} s640 {1-s5-s7:.2f} s768 {s7:.3f} avg {lat:.2f} | GT {r_gt} | learned 2-target {r_ln} | learned 1-target {r_l1} | null {nl_}")
