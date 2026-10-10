"""Routes over RT-DETRv2-R50's native (decoder depth d, query count K) grid on the subset.
Needs probe.npz + *_perimage.json (per-image AP for the grid points). One thread.
Cost model (T4, est.): paper Table 5, R50: 6 layers 9.3 ms, -0.45 ms per removed layer; K=100 scales each decoder
layer by the CPU ratio dec100/dec300 (est.); branch penalty +0.3 ms for an If (T4-envelope.md: +0.25-0.5)."""
import os, sys, json, io, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
rng = np.random.default_rng(0)
z = np.load(sys.argv[1]); n = int(z["n"]); sig = json.loads(str(z["signals"])); T = json.loads(str(z["timings"]))
ids = [r["id"] for r in sig]; pi = json.load(open(sys.argv[1].replace(".npz", "_perimage.json")))
dense = json.load(open(sys.argv[1].replace(".npz", "_dense.json")))
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO("/data/datasets/coco/annotations/instances_val2017.json")
dets = {k: z[k] for k in z.files if k.startswith("d")}
def to_list(a):
    return [{"image_id": int(r[0]), "category_id": int(r[1]), "score": float(r[2]), "bbox": [float(v) for v in r[3:7]]} for r in a]
def AP(a):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(to_list(a)); E = COCOeval(gt, dt, "bbox"); E.params.imgIds = ids; E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0]
def mix(choice):   # choice: image_id -> key
    return np.concatenate([dets[k][dets[k][:, 0] == i] for i, k in choice.items()])
ratioK = T["dec100"] / T["dec300"]
def cost(d, K, branch=0.0):
    return 9.3 - 0.45 * (6 - d) * (1 if K == 300 else 1) - (0.45 * d * (1 - ratioK) if K == 100 else 0) + branch
print(f"n={n}  CPU share backbone/encoder/dec300/dec100 = {T['backbone']:.0f}/{T['encoder']:.0f}/{T['dec300']:.0f}/{T['dec100']:.0f} s; dec100/dec300={ratioK:.2f}")
print("dense grid (AP on subset, T4 ms est. from paper table):")
for K in (300, 100):
    print(" K=%d: " % K + "  ".join(f"d{d}:{dense[f'd{d}_K{K}']['AP']:.4f}@{cost(d,K):.2f}" for d in range(1, 7)))
gtc = {i: len(gt.getAnnIds(imgIds=[i], iscrowd=None)) for i in ids}
cheap, full = sys.argv[2], sys.argv[3]          # e.g. d3_K300 d6_K300
pc = np.array([pi[cheap].get(str(i), np.nan) for i in ids]); pf = np.array([pi[full].get(str(i), np.nan) for i in ids])
delta = np.nan_to_num(pf - pc)                 # per-image gain of full over cheap (images without GT: 0)
dc, df_ = int(cheap[1]), int(full[1])
print(f"\nroute {cheap} -> {full}: images where full>cheap {np.mean(delta>0):.2f}, cheap>full {np.mean(delta<0):.2f}, equal {np.mean(delta==0):.2f}; mean delta {delta.mean():+.4f}")
# signals (ground-truth free), computed at the cheap depth
iou = np.array([r.get(f"iou{dc}", 1.0) for r in sig]); sc = np.array([r[f"s{dc}"] for r in sig]); nn_ = np.array([r[f"n{dc}"] for r in sig])
cnt = np.array([gtc[i] for i in ids])
from scipy.stats import spearmanr
print(f"Spearman(signal, per-image gain of full): iou {spearmanr(iou, delta)[0]:+.3f}  score {spearmanr(sc, delta)[0]:+.3f}  n>0.3 {spearmanr(nn_, delta)[0]:+.3f}  GT count {spearmanr(cnt, delta)[0]:+.3f}")
print(f"Spearman(signal, GT count): iou {spearmanr(iou, cnt)[0]:+.3f}  score {spearmanr(sc, cnt)[0]:+.3f}  n>0.3 {spearmanr(nn_, cnt)[0]:+.3f}")
# out-of-fold ridge on [iou, score, n] -> delta  (5 folds)
X = np.stack([iou, sc, np.log1p(nn_)], 1); X = (X - X.mean(0)) / (X.std(0) + 1e-9); Xb = np.c_[X, np.ones(len(X))]
pred = np.zeros(len(X)); folds = np.array_split(rng.permutation(len(X)), 5)
for f in folds:
    tr = np.setdiff1d(np.arange(len(X)), f); w = np.linalg.solve(Xb[tr].T @ Xb[tr] + 1.0 * np.eye(4), Xb[tr].T @ delta[tr]); pred[f] = Xb[f] @ w
print(f"OOF ridge Spearman with gain {spearmanr(pred, delta)[0]:+.3f}")
base_c, base_f = dense[cheap]["AP"], dense[full]["AP"]
print(f"\nshare -> AP: oracle | share-null (mean of 5 draws) | GT-count rule | learned (OOF ridge) | iou rule | est. T4 ms (one If, +0.3)")
for share in (0.25, 0.5, 0.75):
    m = int(round(share * n))
    def route(order):
        sel = set(np.array(ids)[order[:m]]); return AP(mix({i: (full if i in sel else cheap) for i in ids}))
    orc = route(np.argsort(-delta)); nul = np.mean([route(rng.permutation(n)) for _ in range(5)])
    cnt_r = route(np.argsort(-cnt + rng.random(n) * 1e-3)); lrn = route(np.argsort(-pred)); iou_r = route(np.argsort(iou))
    ms = cost(dc, 300) * (1 - share) + cost(df_, 300) * share + 0.3
    print(f"{share:.2f} -> {orc:.4f} | {nul:.4f} | {cnt_r:.4f} | {lrn:.4f} | {iou_r:.4f} | {ms:.2f} ms   (dense cheap {base_c:.4f}@{cost(dc,300):.2f}, full {base_f:.4f}@{cost(df_,300):.2f})")
