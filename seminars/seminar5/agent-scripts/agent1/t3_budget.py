# three-scale learned routes at several T4 average-latency budgets against the T4 bar F_real(L)+0.003
import numpy as np, mixlib as M, feats as F
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dumpml_yolo26m_768_coco")]
ids, ns, nm, nl, minA = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
pn_s = F.oof_ridge(Xn, np.log1p(ns), 10); pn_sm = F.oof_ridge(Xn, np.log1p(ns + nm), 10); pn_l = F.oof_ridge(Xn, np.log1p(nl), 10)
rng = np.random.RandomState(11)
def route(hi, lo, s5, s7):
    ch = np.ones(I, int); o = np.lexsort((rng.rand(I), -hi)); ch[o[:int(round(s7 * I))]] = 2
    rest = np.where(ch == 1)[0]; o2 = rest[np.lexsort((rng.rand(len(rest)), lo[rest]))]; ch[o2[:int(round(s5 * I))]] = 0; return ch
def run(T, tag):
    print(f"== branch ms {T} ({tag})")
    for s5 in (0.0, 0.1, 0.2, 0.3, 0.4):
        for s7 in (0.0, 0.1, 0.2, 0.3, 0.4):
            if s5 + s7 > 0.8 or (s5 == 0 and s7 == 0): continue
            L = s5 * T[0] + (1 - s5 - s7) * T[1] + s7 * T[2]; bar = 0.5261 + 0.0102 * (L - 5.05) + 0.003
            a = M.score(srcs, route(pn_s, pn_sm, s5, s7))[0]; b = M.score(srcs, route(pn_s, pn_sm - 0.5 * pn_l, s5, s7))[0]
            print(f"s512 {s5:.1f} s768 {s7:.1f} avg {L:.2f} bar {bar:.4f} | learned {a:.4f} ({a-bar:+.4f}) | learned-v2 {b:.4f} ({b-bar:+.4f})")
run(np.array([3.47, 5.05, 7.12]), "measured 512/768 in the 768-input engine, 640 = dense real")
