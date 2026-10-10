# Is capacity routable by scene scale? M/L per image with L placed by the thumbnail-predicted LARGE-object count (agent 4's axis)
# vs by total count (seminar 4 / agent 7) vs the share null; and the exchange with L placed by the large count.
import sys; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1")
import numpy as np, mixlib as M, feats as F
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco")]
ids, ns, nm, nl, _ = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
p_l = F.oof_ridge(Xn, np.log1p(nl), 10); p_n = F.oof_ridge(Xn, np.log1p(ns + nm + nl), 10); p_sm = F.oof_ridge(Xn, np.log1p(ns + nm), 10)
p_lf = F.oof_ridge(Xn, nl / np.maximum(ns + nm + nl, 1), 10)
rng = np.random.RandomState(21)
def top(score, idx, n): return idx[np.lexsort((rng.rand(len(idx)), -score[idx]))][:n]
for sL in (0.16, 0.31, 0.42):
    n = int(round(sL * I)); allidx = np.arange(I); res = {}
    for nm_, sc in (("large count", p_l), ("large fraction", p_lf), ("total count", p_n), ("GT large count", np.log1p(nl) + 1e-3 * (ns + nm))):
        ch = np.ones(I, int); ch[top(sc, allidx, n)] = 2; res[nm_] = M.score(srcs, ch)[0]
    nu = []
    for _ in range(3): ch = np.ones(I, int); ch[rng.choice(I, n, replace=False)] = 2; nu.append(M.score(srcs, ch)[0])
    print(f"M/L, L share {sL}: " + " | ".join(f"{k} {v:.4f} ({v-np.mean(nu):+.4f})" for k, v in res.items()) + f" | null {np.mean(nu):.4f}", flush=True)
for q, qL in ((0.4, 0.2), (0.5, 0.3), (0.5, 0.4)):
    ch = np.ones(I, int); o = np.lexsort((rng.rand(I), p_sm)); ch[o[:int(round(q * I))]] = 0; rest = np.where(ch == 1)[0]
    for nm_, sc in (("large count", p_l), ("total count", p_n)):
        c2 = ch.copy(); c2[top(sc, rest, int(round(qL * I)))] = 2; a = M.score(srcs, c2)
        t = q * 3.47 + (1 - q - qL) * 5.05 + qL * 6.55; Fr = 0.5261 + 0.0102 * (t - 5.05)
        print(f"exchange q512 {q} qL {qL}, L by {nm_}: {a[0]:.4f} S/M/L {a[1]}/{a[2]}/{a[3]} at {t:.2f} ms real (opt) | vs bar {a[0]-Fr-0.003:+.4f}", flush=True)
