# Round 2: does the capacity half of agent 7's exchange carry routing? {M@512, M@640, L@640} with the L share placed by the
# thumbnail count router vs placed at random among the non-shrunk images; against the linear front, the share null and the
# predicted dense resolution envelope (M and L rescaled; prediction from M's three measured scales, latency pixel model).
import sys; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1")
import numpy as np, mixlib as M, feats as F
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco")]
ids, ns, nm, nl, _ = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
p_sm = F.oof_ridge(Xn, np.log1p(ns + nm), 10); p_n = F.oof_ridge(Xn, np.log1p(ns + nm + nl), 10)
g_sm = np.log1p(ns + nm) + 1e-3 * nl; g_n = np.log1p(ns + nm + nl)
rng = np.random.RandomState(11)
BASES = {"opt": (3.47, 5.05, 6.55), "pes": (3.60, 5.23, 6.73), "directfed+0.65": (3.47, 5.70, 7.20)}
c = np.polyfit(np.log(np.array([512, 640, 768]) / 640), [0.5063, 0.5261, 0.5196], 2); af = 0.661 / 5.05
def dense_curve(t, ap640, t640):  # best rescaled dense model of this family member at latency t (pred.)
    fr = t / t640; p2 = (fr - af) / (1 - af)
    if p2 <= 0.3 or p2 > 1.44: return -1
    return ap640 + np.polyval(c, np.log(np.sqrt(p2))) - np.polyval(c, 0.0)
def envelope(t): return max(dense_curve(t, 0.5261, 5.05), dense_curve(t, 0.5417, 6.55))
def route(q, qL, lo, hi, Lmode):
    ch = np.ones(I, int); o = np.lexsort((rng.rand(I), lo)); ch[o[:int(round(q * I))]] = 0
    rest = np.where(ch == 1)[0]; nL = int(round(qL * I))
    if Lmode == "random": pick = rng.choice(rest, nL, replace=False)
    else: pick = rest[np.lexsort((rng.rand(len(rest)), -hi[rest]))][:nL]
    ch[pick] = 2; return ch
def null(q, qL, k=3):
    v = []
    for _ in range(k):
        p = rng.permutation(I); ch = np.ones(I, int); ch[p[:int(round(q * I))]] = 0; ch[p[int(round(q * I)):int(round(q * I)) + int(round(qL * I))]] = 2; v.append(M.score(srcs, ch)[0])
    return np.mean(v)
print("alone 512/640/L:", [M.score(srcs, np.full(I, k)) for k in range(3)], flush=True)
print("pred. envelope at real-basis ms:", {t: round(envelope(t), 4) for t in (4.26, 4.42, 4.6, 4.8, 5.05, 5.3, 5.5)}, flush=True)
for q in (0.3, 0.4, 0.5, 0.6):
    for qL in (0.0, 0.1, 0.2, 0.3, 0.4):
        if q + qL > 0.95: continue
        a = M.score(srcs, route(q, qL, p_sm, p_n, "count")); r = np.mean([M.score(srcs, route(q, qL, p_sm, p_n, "random"))[0] for _ in range(3)])
        g = M.score(srcs, route(q, qL, g_sm, g_n, "count"))[0]; n0 = null(q, qL) if qL in (0.0, 0.2, 0.4) else float("nan")
        s = f"q512 {q:.1f} qL {qL:.1f} | learned {a[0]:.4f} S/M/L {a[1]}/{a[2]}/{a[3]} | L placed at random {r:.4f} (count-random {a[0]-r:+.4f}) | GT {g:.4f} | null {n0:.4f}"
        for b, T in BASES.items():
            t = q * T[0] + (1 - q - qL) * T[1] + qL * T[2]; Fr = 0.5261 + 0.0102 * (t - 5.05)
            s += f" | {b}: {t:.2f} ms bar {Fr+0.003:.4f} m {a[0]-Fr-0.003:+.4f}"
        t = q * 3.47 + (1 - q - qL) * 5.05 + qL * 6.55; s += f" | vs pred. envelope {a[0]-envelope(t):+.4f}"
        print(s, flush=True)
