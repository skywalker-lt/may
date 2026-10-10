# label-file (bbox-area) counts, so the train2017 target is defined the same way on val; compare with COCO segment-area classes
import os, sys, numpy as np, cv2
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1"); import feats as F
def lab_areas(split, iid):
    p = f"/data/datasets/coco/labels/{split}/{iid:012d}.txt"
    if not os.path.exists(p): return np.zeros(0)
    im = cv2.imread(f"/data/datasets/coco/images/{split}/{iid:012d}.jpg"); h, w = im.shape[:2]
    return np.array([float(v.split()[3]) * w * float(v.split()[4]) * h for v in open(p) if len(v.split()) >= 5])
def counts(areas, t1, t2):
    return np.array([(a < t1).sum() for a in areas]), np.array([((a >= t1) & (a < t2)).sum() for a in areas]), np.array([(a >= t2).sum() for a in areas])
if __name__ == "__main__":
    ids, ns, nm, nl, _ = F.gt_counts(); A = [lab_areas("val2017", int(i)) for i in ids]
    np.save("/data/tmp/ds-yolo/seminar5/work/agent1/r2/val_lab_areas.npy", np.array(A, dtype=object), allow_pickle=True)
    for k in (1.0, 1.3, 1.5, 1.7, 2.0):
        s, m, l = counts(A, k * 32**2, k * 96**2)
        y0, y1 = np.log1p(ns + nm), np.log1p(s + m)
        from scipy.stats import spearmanr
        print(f"bbox thresholds x{k}: corr log1p(S+M) {np.corrcoef(y0, y1)[0,1]:.4f} spearman {spearmanr(y0, y1)[0]:.4f} | totals S {s.sum()} vs {int(ns.sum())}, M {m.sum()} vs {int(nm.sum())}, L {l.sum()} vs {int(nl.sum())}")
