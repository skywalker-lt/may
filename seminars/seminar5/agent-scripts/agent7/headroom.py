"""Headroom of mixtures of whole models (seminar-4 convention): proxy oracle / share null / GT count rule / learned router."""
import os, sys, time, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from mixlib import mix_ap, load, img_ids, gt_counts
from scipy.stats import spearmanr
ids = np.array(img_ids()); z = np.load("/data/tmp/ds-yolo/seminar5/inputs/dumps/val2017_stem_pooled.npz")
order = np.argsort(z["image_id"]); assert (z["image_id"][order] == ids).all()
Xn = z["n320"][order].astype(np.float64)
n_all, n_small, med_area = gt_counts(); I = len(ids)
def ridge_oof(X, y, lam=10.0, k=5, seed=0):
    rng = np.random.RandomState(seed); f = rng.randint(0, k, len(y)); p = np.zeros(len(y))
    for j in range(k):
        tr, te = f != j, f == j; mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        A = (X[tr] - mu) / sd; ym = y[tr].mean(); w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]) / 100, A.T @ (y[tr] - ym))
        p[te] = (X[te] - mu) / sd @ w + ym
    return p
def top_share(score, s):
    a = np.zeros(I, int); k = int(round(s * I)); a[np.argsort(-score)[:k]] = 1; return a
def null(names, s, draws=3):
    r = []
    for d in range(draws):
        a = np.zeros(I, int); a[np.random.RandomState(100 + d).permutation(I)[:int(round(s * I))]] = 1; r.append(mix_ap(names, a))
    return np.mean(r)
prox = lambda n: np.nan_to_num(load(n)["proxy"], nan=0.0)
cnt_pred_small = ridge_oof(Xn, np.log1p(n_small)); cnt_pred_all = ridge_oof(Xn, np.log1p(n_all))
print("thumbnail->log small count spearman", round(spearmanr(cnt_pred_small, n_small)[0], 3), " ->log count", round(spearmanr(cnt_pred_all, n_all)[0], 3), flush=True)
def pair(a, b, shares, tag):
    """share = fraction of images sent to b."""
    g = prox(b) - prox(a); gain_pred = ridge_oof(Xn, g)
    print(f"== {tag}: a={a} ({load(a)['stats'][0]:.4f}) b={b} ({load(b)['stats'][0]:.4f})", flush=True)
    for nm, v in (("small count", n_small), ("count", n_all), ("median area", -med_area), ("pred small count", cnt_pred_small), ("pred gain", gain_pred)):
        print(f"   spearman(proxy gain b-a, {nm}) = {spearmanr(g, v)[0]:+.3f}")
    for s in shares:
        t0 = time.time(); names = [a, b]
        row = {"oracle(proxy)": mix_ap(names, top_share(g, s)), "null": null(names, s), "GT small-count rule": mix_ap(names, top_share(n_small + 1e-3 * n_all, s)),
               "GT count rule": mix_ap(names, top_share(n_all, s)), "learned n320 small count": mix_ap(names, top_share(cnt_pred_small, s)),
               "learned n320 gain": mix_ap(names, top_share(gain_pred, s)), "learned n320 -count": mix_ap(names, top_share(-cnt_pred_all, s))}
        print(f"   share {s:.2f}: " + " | ".join(f"{k} {v:.4f}" for k, v in row.items()) + f"  ({time.time()-t0:.0f}s)", flush=True)
if __name__ == "__main__":
    which = sys.argv[1]
    if which == "iso":
        pair("dumpml_yolo26m_512_coco", "dumpml_yolo26s_768_coco", [0.3, 0.5], "iso-latency ~3.7 ms T4: M@512 vs S@768")
        pair("dump_yolo26l_coco", "dumpml_yolo26m_768_coco", [0.2, 0.4], "iso-latency ~6.9 ms T4: L@640 vs M@768")
    if which == "fam":
        pair("dumpml_yolo26m_coco", "dump_yolov12m_sdpa_coco", [0.2, 0.5], "families ~5.4 ms: YOLO26-M vs YOLOv12-M")
        pair("dumpml_yolo26m_coco", "dump_yolo11m_coco", [0.2, 0.5], "families ~5.3 ms: YOLO26-M vs YOLO11-M")
