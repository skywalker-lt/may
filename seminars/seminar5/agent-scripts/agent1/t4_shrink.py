# shrink-or-magnify: large 512 shares, small 768 shares; learned (n320 thumbnail) vs GT-rule bound vs null, against the T4 bar
import numpy as np, mixlib as M, feats as F
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dumpml_yolo26m_768_coco")]
ids, ns, nm, nl, minA = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
pn_s = F.oof_ridge(Xn, np.log1p(ns), 10); pn_sm = F.oof_ridge(Xn, np.log1p(ns + nm), 10)
pn_frac = F.oof_ridge(Xn, (ns + nm) / np.maximum(ns + nm + nl, 1), 10); pm_sm = F.oof_ridge(Xm, np.log1p(ns + nm), 10)
gt_hi = np.log1p(ns) + 1e-3 * nm; gt_lo = np.log1p(ns + nm) + 1e-3 * nl
rng = np.random.RandomState(5); T = np.array([3.47, 5.05, 7.12])
def route(hi, lo, s5, s7):
    ch = np.ones(I, int); o = np.lexsort((rng.rand(I), -hi)); ch[o[:int(round(s7 * I))]] = 2
    rest = np.where(ch == 1)[0]; o2 = rest[np.lexsort((rng.rand(len(rest)), lo[rest]))]; ch[o2[:int(round(s5 * I))]] = 0; return ch
def null(s5, s7, k=3):
    v = []
    for _ in range(k):
        p = rng.permutation(I); ch = np.ones(I, int); n7 = int(round(s7 * I)); ch[p[:n7]] = 2; ch[p[n7:n7 + int(round(s5 * I))]] = 0; v.append(M.score(srcs, ch)[0])
    return np.mean(v)
print("no-route reference: 512 / 640 / 768 =", [M.score(srcs, np.full(I, k))[0] for k in range(3)])
for s7 in (0.0, 0.05, 0.1, 0.15):
    for s5 in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        if s5 + s7 > 0.95: continue
        L = s5 * T[0] + (1 - s5 - s7) * T[1] + s7 * T[2]; F_ = 0.5261 + 0.0102 * (L - 5.05); bar = F_ + 0.003
        a = M.score(srcs, route(pn_s, pn_sm, s5, s7)); c = M.score(srcs, route(pn_s, pn_frac, s5, s7))[0]
        g = M.score(srcs, route(gt_hi, gt_lo, s5, s7))[0]; n0 = null(s5, s7)
        print(f"s512 {s5:.2f} s768 {s7:.2f} avg {L:.2f} front {F_:.4f} bar {bar:.4f} | learned {a[0]:.4f} ({a[0]-F_:+.4f} vs front, {a[0]-n0:+.4f} vs null) S/M/L {a[1]}/{a[2]}/{a[3]} | learned-frac {c:.4f} | GT {g:.4f} | null {n0:.4f}", flush=True)
