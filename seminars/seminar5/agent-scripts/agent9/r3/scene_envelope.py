"""Round 3 (agent 9): can a GT-free scene partition choose the dense OPERATING POINT (model, input scale) per scene
better than the dense envelope? Cross-fitted: k-means (or count bins) fit on one half of val2017, each cluster's operating
point chosen on that half by summed per-image proxy AP under a Lagrangian latency price, applied to the other half; then
exact pycocotools AP of the full mixture (agent 7's mixlib, caches read-only). Costs: agent 7's T4 unset basis
(M@512/640/768 and L@640 measured, the rest est. by pixel scaling), router 0.18 ms on every image."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7/r2")
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent9")
import menus as M7                      # dense points, envelope, front, router cost, out-of-fold count prediction
from mixlib import mix_ap, load
from common import kmeans, assign as kassign
from headroom import n_all
I = 5000; keys = sorted(M7.pts, key=lambda k: M7.pts[k][1]); names = [M7.pts[k][0] for k in keys]
cost = np.array([M7.pts[k][1] for k in keys]); P = np.stack([np.nan_to_num(load(n)["proxy"], nan=0.0) for n in names], 1)
z = np.load("/data/tmp/ds-yolo/seminar5/inputs/dumps/val2017_stem_pooled.npz"); ids = np.array(load(names[0])["imgIds"])
o = np.argsort(z["image_id"]); assert (z["image_id"][o] == ids).all()
Feat = {k: z[k][o].astype(np.float64) for k in ("n320", "m640")}
for k in Feat: Feat[k] = (Feat[k] - Feat[k].mean(0)) / (Feat[k].std(0) + 1e-6)
def choose(lab_tr, tr, K, target):
    Pc = np.zeros((K, len(keys))); Nc = np.bincount(lab_tr, minlength=K).astype(float)
    np.add.at(Pc, lab_tr, P[tr])
    best = None
    for lam in np.concatenate([np.linspace(0, 0.2, 801), [1e3]]):
        pick = np.argmax(Pc - lam * Nc[:, None] * cost[None], 1); L = (Nc * cost[pick]).sum() / Nc.sum() + M7.R
        if L <= target + 1e-9 and (best is None or Pc[np.arange(K), pick].sum() > best[0]): best = (Pc[np.arange(K), pick].sum(), pick)
    return best[1]
def crossfit(part, K, target, seed=0):
    rng = np.random.RandomState(seed); fold = rng.rand(I) < 0.5; a = np.zeros(I, int)
    for f in (True, False):
        tr, te = fold == f, fold != f
        if part in Feat:
            c, _ = kmeans(Feat[part][tr], K, np.random.default_rng(seed)); lab = kassign(Feat[part], c)
        elif part == "predcount":
            q = np.quantile(M7.cnt_pred_all[tr], np.linspace(0, 1, K + 1)[1:-1]); lab = np.searchsorted(q, M7.cnt_pred_all)
        elif part == "GTcount":
            q = np.quantile(n_all[tr], np.linspace(0, 1, K + 1)[1:-1]); lab = np.searchsorted(q, n_all + 1e-6 * np.random.RandomState(1).rand(I))
        elif part == "random":
            lab = np.random.RandomState(50 + seed).randint(0, K, I)
        pick = choose(lab[tr], np.where(tr)[0], K, target); a[te] = pick[lab[te]]
    return a
def report(tag, a):
    L = cost[a].mean() + M7.R; ap = mix_ap(names, a); e = M7.env(L); sh = np.bincount(a, minlength=len(keys)) / I
    mix = " ".join(f"{keys[j]}:{sh[j]:.2f}" for j in range(len(keys)) if sh[j] > 0)
    nl = mix_ap(names, np.random.RandomState(7).permutation(a))
    print(f"{tag:28s} avg {L:.2f} ms AP {ap:.4f} | -env {ap-e:+.4f} (env {e:.4f}) | -F {ap-M7.F(L):+.4f} | -null {ap-nl:+.4f} | {mix}", flush=True)
if __name__ == "__main__":
    for target in (4.70, 5.36):
        print(f"== budget {target} ms (T4 unset basis, est.)", flush=True)
        for part, Ks in (("m640", (4, 8, 16)), ("n320", (4, 8, 16)), ("predcount", (4, 8)), ("GTcount", (4, 8)), ("random", (8,))):
            for K in Ks: report(f"{part} K={K}", crossfit(part, K, target))
