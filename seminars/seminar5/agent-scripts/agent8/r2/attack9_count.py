"""Round 2, attack on direction 9: are agent 9's m640/n320 K=4 scene clusters a coarse object-count code?
eta^2 of log(1+GT count) by cluster, cluster mean count, and the rank correlation between cluster mean count and the
cluster's L-M gain (agent 9's ranking variable). Uses agent 9's kmeans/feats read-only."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"; sys.dont_write_bytecode = True
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent9"); os.chdir("/data/tmp/ds-yolo/seminar5/work/agent9")
from common import *
from scipy.stats import spearmanr
M = load('dumpml_yolo26m_coco'); L = load('dump_yolo26l_coco'); F = feats(M); n = len(M['imgIds']); Y = gt_matrix(M); cnt = Y.sum(1)
y = np.log1p(cnt)
for fk in ("m640", "n320"):
    for K in (4, 8, 16):
        e2 = []; rhos = []
        for sd in range(3):
            rng = np.random.default_rng(500 + sd); c, lab = kmeans(F[fk], K, rng)
            mu = np.array([y[lab == k].mean() for k in range(K)]); e2.append(((mu[lab] - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())
            g = []; mc = []
            for k in range(K):
                m = lab == k
                if m.sum() > 20: g.append(ap_rec(L, m) - ap_rec(M, m)); mc.append(cnt[m].mean())
            rhos.append(spearmanr(g, mc)[0])
            if sd == 0: print(f"{fk} K={K}: cluster sizes {np.bincount(lab)} mean count {np.round([cnt[lab==k].mean() for k in range(K)],1)} L-M gain {np.round(g,4)}", flush=True)
        print(f"{fk} K={K}: eta^2(log count | cluster) {np.mean(e2):.3f} (3 seeds {np.round(e2,3)}); Spearman(cluster mean count, cluster L-M gain) {np.mean(rhos):+.2f} ({np.round(rhos,2)})", flush=True)
