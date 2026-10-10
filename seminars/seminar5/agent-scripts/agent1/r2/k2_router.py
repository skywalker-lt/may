# Gate K2: router fitted on train2017 (24k random images, n320, label-bbox target log1p(S+M) at x1.7 thresholds), thresholds at
# train quantiles; scored on val2017 against the round-1 out-of-fold val router. Saves the router for the RS-2 engine graph.
import sys; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1/r2")
import numpy as np, mixlib as M, feats as F
d = np.load("/data/tmp/ds-yolo/seminar5/work/agent1/r2/train_n320.npz", allow_pickle=True)
Xt = d["n320"].astype(np.float64); At = d["areas"]; k = 1.7
yt = np.array([np.log1p((a < k * 96**2).sum()) for a in At])
mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Z = (Xt - mu) / sd; lam = 10
w = np.linalg.solve(Z.T @ Z + lam * len(Z) / 100 * np.eye(Z.shape[1]), Z.T @ (yt - yt.mean())); b = yt.mean()
pt = Z @ w + b
ids, ns, nm, nl, _ = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
pv = ((Xn - mu) / sd) @ w + b; poof = F.oof_ridge(Xn, np.log1p(ns + nm), 10)
from scipy.stats import spearmanr
print(f"train n={len(yt)}; train in-sample spearman {spearmanr(pt, yt)[0]:.3f}; val spearman with GT log1p(S+M) {spearmanr(pv, np.log1p(ns+nm))[0]:.3f} (OOF val router {spearmanr(poof, np.log1p(ns+nm))[0]:.3f})")
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco")]
for s in (0.4, 0.5, 0.6, 0.7):
    thr = np.quantile(pt, s); ch = (pv >= thr).astype(int); real = 1 - ch.mean()
    o = np.argsort(poof, kind="stable"); ch2 = np.ones(I, int); ch2[o[:int(round(real * I))]] = 0
    a, a2 = M.score(srcs, ch), M.score(srcs, ch2)
    print(f"target share {s:.2f}: train-quantile threshold {thr:.4f} -> realised val 512 share {real:.3f} | train router AP {a[0]:.4f} S/M/L {a[1]}/{a[2]}/{a[3]} | OOF val router at the same realised share {a2[0]:.4f} | diff {a[0]-a2[0]:+.4f}", flush=True)
    if s == 0.5: np.savez("/data/tmp/ds-yolo/seminar5/work/agent1/r2/router_train.npz", mu=mu, sd=sd, w=w, b=b, thr=thr, share=s)
