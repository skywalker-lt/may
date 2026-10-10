# Attack on direction 9: how much of a GT-free k-means scene partition (agent 9's recipe: standardised pooled stem
# feature, k-means++) is the object-count axis? AUC of log GT count and of the thumbnail-predicted count for the K=2
# cluster label; mean counts per cluster for K=2 and K=4; agreement of the K=2 partition with a median split of predicted count.
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, pickle, numpy as np
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent9"); from common import kmeans, assign
from scipy.stats import mannwhitneyu, spearmanr
F = pickle.load(open("/data/tmp/ds-yolo/seminar5/work/agent10/features.pkl", "rb"))
z = np.load("/data/tmp/ds-yolo/seminar5/inputs/dumps/val2017_stem_pooled.npz"); assert (z["image_id"] == F["ids"]).all()
cnt, pc = F["cnt"], F["pred_cnt"]
def auc(lab, v):
    a, b = v[lab == 1], v[lab == 0]; u = mannwhitneyu(a, b).statistic; return u / (len(a) * len(b))
for fk in ["m640", "n320"]:
    X = z[fk].astype(np.float64); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    for K in (2, 4):
        aucs, agree, means = [], [], []
        for sd in range(3):
            rng = np.random.default_rng(500 + sd); c, lab = kmeans(X, K, rng)
            m = [cnt[lab == k].mean() for k in range(K)]; o = np.argsort(m); r = np.argsort(o)[lab]  # rank clusters by mean count
            means.append(sorted(m))
            if K == 2:
                aucs.append((auc(r, np.log1p(cnt)), auc(r, pc)))
                med = (pc > np.median(pc)).astype(int); agree.append(max((med == r).mean(), (med != r).mean()))
            else:
                aucs.append((spearmanr(r, cnt)[0], spearmanr(r, pc)[0]))
        print(f"{fk} K={K}: per-cluster mean GT count {np.round(np.mean(means,0),1)}; "
              + (f"AUC(log GT count) {np.mean([a[0] for a in aucs]):.3f}, AUC(pred count) {np.mean([a[1] for a in aucs]):.3f}, agreement with median split of predicted count {np.mean(agree):.3f}" if K == 2
                 else f"Spearman(cluster rank by count, GT count) {np.mean([a[0] for a in aucs]):.3f}, with predicted count {np.mean([a[1] for a in aucs]):.3f}"))
