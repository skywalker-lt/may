"""Agent 10 round 1: per-image exit route between a cheap exit and the full decoder, on probe npz + eval_depth outputs.
Rows: oracle (per-image AP proxy, upper bound), share-matched random null (10 draws, mean and sd), GT-count rule,
convergence rule (low top-30 IoU between the last two exits -> full), learned (5-fold OOF ridge on the GT-free signals),
and the base's own dense depth envelope at the routed point's est. T4 latency (cost model given on the command line).
Usage: analyze_route.py probe.npz cheap_key full_key full_ms ms_per_layer branch_ms   ONE thread."""
import os, sys, json, io, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.stats import spearmanr
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
rng = np.random.default_rng(0)
f, cheap, full = sys.argv[1], sys.argv[2], sys.argv[3]
full_ms, per_layer, branch = map(float, sys.argv[4:7])
z = np.load(f); n = int(z["n"]); sig = json.loads(str(z["signals"])); ids = [r["id"] for r in sig]
pi = json.load(open(f.replace(".npz", "_perimage.json"))); dense = json.load(open(f.replace(".npz", "_dense.json")))
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO("/data/datasets/coco/annotations/instances_val2017.json")
dets = {k: z[k] for k in z.files if k.startswith("d")}
byimg = {k: {i: v[v[:, 0] == i] for i in ids} for k, v in dets.items() if k in (cheap, full)}
def to_list(a): return [{"image_id": int(r[0]), "category_id": int(r[1]), "score": float(r[2]), "bbox": [float(v) for v in r[3:7]]} for r in a]
def AP(a):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(to_list(a)); E = COCOeval(gt, dt, "bbox"); E.params.imgIds = ids; E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0]
def mix(sel): return np.concatenate([byimg[full if i in sel else cheap][i] for i in ids])
maxd = max(int(k[1:].split("_")[0]) for k in dense)
def cost(d): return full_ms - per_layer * (maxd - d)
dc, dfull = int(cheap[1:].split("_")[0]), int(full[1:].split("_")[0])
print(f"{f}: n={n}; dense subset AP by exit (est. T4 ms):", "  ".join(f"{k}:{v['AP']:.4f}@{cost(int(k[1:].split('_')[0])):.2f}" for k, v in sorted(dense.items()) if k.endswith("K300")))
gtc = np.array([len(gt.getAnnIds(imgIds=[i], iscrowd=False)) for i in ids])
pc = np.array([pi[cheap].get(str(i), np.nan) for i in ids]); pf = np.array([pi[full].get(str(i), np.nan) for i in ids])
delta = np.nan_to_num(pf - pc)
print(f"per-image AP gain {full} over {cheap}: >0 {np.mean(delta>0):.2f}, <0 {np.mean(delta<0):.2f}, =0 {np.mean(delta==0):.2f}; mean {delta.mean():+.4f}; "
      f"top-25% of images carry {np.sort(delta)[::-1][:n//4].sum()/max(delta[delta>0].sum(),1e-9):.2f} of the positive gain")
iou = np.array([r.get(f"iou{dc}", 1.0) for r in sig]); sc = np.array([r[f"s{dc}"] for r in sig]); nn_ = np.array([r[f"n{dc}"] for r in sig])
print(f"Spearman with gain: conv-IoU {spearmanr(iou, delta)[0]:+.3f}  score {spearmanr(sc, delta)[0]:+.3f}  n>0.3 {spearmanr(nn_, delta)[0]:+.3f}  GT count {spearmanr(gtc, delta)[0]:+.3f}")
print(f"Spearman with GT count: conv-IoU {spearmanr(iou, gtc)[0]:+.3f}  score {spearmanr(sc, gtc)[0]:+.3f}  n>0.3 {spearmanr(nn_, gtc)[0]:+.3f}")
# partial: IoU vs gain after removing rank-linear dependence on GT count
def rk(x): return np.argsort(np.argsort(x)).astype(float)
def resid(y, x): X = np.c_[rk(x), np.ones(len(x))]; return rk(y) - X @ np.linalg.lstsq(X, rk(y), rcond=None)[0]
print(f"partial Spearman (control GT count): conv-IoU vs gain {np.corrcoef(resid(iou, gtc), resid(delta, gtc))[0,1]:+.3f}")
X = np.stack([iou, sc, np.log1p(nn_)], 1); X = (X - X.mean(0)) / (X.std(0) + 1e-9); Xb = np.c_[X, np.ones(n)]
pred = np.zeros(n)
for fo in np.array_split(rng.permutation(n), 5):
    tr = np.setdiff1d(np.arange(n), fo); w = np.linalg.solve(Xb[tr].T @ Xb[tr] + np.eye(4), Xb[tr].T @ delta[tr]); pred[fo] = Xb[fo] @ w
print(f"OOF ridge Spearman with gain {spearmanr(pred, delta)[0]:+.3f}")
dkeys = sorted([k for k in dense if k.endswith("K300")], key=lambda k: int(k[1:].split("_")[0]))
pts = [(cost(int(k[1:].split("_")[0])), dense[k]["AP"]) for k in dkeys]
def env(t):  # upper concave hull of dense exits (a random route realises any chord), no branch penalty
    best = max(a for c, a in pts if c <= t + 1e-9) if any(c <= t for c, _ in pts) else np.nan
    for (c1, a1) in pts:
        for (c2, a2) in pts:
            if c1 < t < c2: best = max(best, a1 + (a2 - a1) * (t - c1) / (c2 - c1))
    return best
print("share | oracle | null mean (sd) | GT-count | conv-IoU | learned | est. T4 ms | dense-exit envelope at ms | learned - null | learned - env")
for share in (0.25, 0.5, 0.75):
    m = int(round(share * n)); A = np.array(ids)
    r = lambda order: AP(mix(set(A[order[:m]])))
    orc = r(np.argsort(-delta)); nulls = [r(rng.permutation(n)) for _ in range(10)]
    cr = r(np.argsort(-gtc + rng.random(n) * 1e-3)); ir = r(np.argsort(iou)); lr = r(np.argsort(-pred))
    ms = cost(dc) * (1 - share) + cost(dfull) * share + branch
    print(f"{share:.2f} | {orc:.4f} | {np.mean(nulls):.4f} ({np.std(nulls):.4f}) | {cr:.4f} | {ir:.4f} | {lr:.4f} | {ms:.2f} | {env(ms):.4f} | {lr-np.mean(nulls):+.4f} | {lr-env(ms):+.4f}", flush=True)
