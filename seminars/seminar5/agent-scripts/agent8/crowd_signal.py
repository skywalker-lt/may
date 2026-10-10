"""GT crowding per image, per-GT recall by same-class overlap for o2o (YOLO26) vs NMS heads (YOLOv12-M, YOLO11-M),
and the router signals (o2o head's own detection count; n320 / m640 pooled stem features). CPU, one thread."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import json, numpy as np, pickle
from collections import defaultdict
from scipy.stats import spearmanr
D = "/data/tmp/ds-yolo/seminar5/inputs/dumps/"
W = "/data/tmp/ds-yolo/seminar5/work/agent8/"
gt = json.load(open(D + "instances_val2017.json"))
imgs = sorted(i["id"] for i in gt["images"])
G = defaultdict(list)  # img -> list of (cat, box xyxy, area, crowd)
for a in gt["annotations"]:
    x, y, w, h = a["bbox"]
    G[a["image_id"]].append((a["category_id"], np.array([x, y, x + w, y + h]), a["area"], a["iscrowd"]))

def iou_mat(A, B):
    if len(A) == 0 or len(B) == 0:
        return np.zeros((len(A), len(B)))
    A = np.asarray(A); B = np.asarray(B)
    lt = np.maximum(A[:, None, :2], B[None, :, :2]); rb = np.minimum(A[:, None, 2:], B[None, :, 2:])
    wh = np.clip(rb - lt, 0, None); inter = wh[..., 0] * wh[..., 1]
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)

# per-image crowd statistics and per-GT max same-class IoU
stats = {}
gt_occ = {}  # (img, idx) -> max same-class IoU
for im in imgs:
    anns = [g for g in G[im] if g[3] == 0]
    n = len(anns); ns = sum(1 for g in anns if g[2] < 32 * 32)
    p3 = p5 = 0; occ = np.zeros(n)
    bycat = defaultdict(list)
    for j, g in enumerate(anns):
        bycat[g[0]].append(j)
    maxcat = max((len(v) for v in bycat.values()), default=0)
    for c, idx in bycat.items():
        if len(idx) < 2:
            continue
        M = iou_mat([anns[j][1] for j in idx], [anns[j][1] for j in idx]); np.fill_diagonal(M, 0)
        occ[idx] = M.max(1)
        tri = M[np.triu_indices(len(idx), 1)]
        p3 += int((tri > 0.3).sum()); p5 += int((tri > 0.5).sum())
    stats[im] = dict(n=n, ns=ns, p3=p3, p5=p5, maxcat=maxcat, anycrowd=int(any(g[3] for g in G[im])))
    for j in range(n):
        gt_occ[(im, j)] = occ[j]

def load_dets(f):
    d = defaultdict(list)
    for r in json.load(open(D + f)):
        x, y, w, h = r["bbox"]
        d[r["image_id"]].append((r["category_id"], r["score"], x, y, x + w, y + h))
    return d

models = {"yolo26m(o2o)": "dumpml_yolo26m_coco.json", "yolo26l(o2o)": "dump_yolo26l_coco.json",
          "yolov12m(NMS)": "dump_yolov12m_sdpa_coco.json", "yolo11m(NMS)": "dump_yolo11m_coco.json"}
bins = [(-1, 1e-9, "isolated (0)"), (1e-9, 0.3, "(0,0.3]"), (0.3, 0.5, "(0.3,0.5]"), (0.5, 0.7, "(0.5,0.7]"), (0.7, 1.01, ">0.7")]
out = {}
dets_cache = {}
for mname, f in models.items():
    dets = load_dets(f); dets_cache[mname] = dets
    res = {}
    for thr_score, thr_iou in ((0.001, 0.5), (0.25, 0.5), (0.001, 0.75)):
        hit = defaultdict(lambda: [0, 0])
        for im in imgs:
            anns = [g for g in G[im] if g[3] == 0]
            if not anns:
                continue
            dd = [r for r in dets[im] if r[1] >= thr_score]
            dd.sort(key=lambda r: -r[1]); dd = dd[:100]
            for c in set(g[0] for g in anns):
                gi = [j for j, g in enumerate(anns) if g[0] == c]
                di = [r for r in dd if r[0] == c]
                matched = np.zeros(len(gi), bool)
                if di:
                    M = iou_mat([r[2:] for r in di], [anns[j][1] for j in gi])
                    for k in range(len(di)):  # greedy by score, COCO style
                        cand = np.where(~matched & (M[k] >= thr_iou))[0]
                        if len(cand):
                            matched[cand[np.argmax(M[k, cand])]] = True
                for t, j in enumerate(gi):
                    o = gt_occ[(im, j)]
                    for lo, hi, lab in bins:
                        if lo < o <= hi or (lab.startswith("isolated") and o == 0):
                            hit[lab][0] += matched[t]; hit[lab][1] += 1; break
        res[(thr_score, thr_iou)] = {lab: (hit[lab][0] / max(1, hit[lab][1]), hit[lab][1]) for _, _, lab in bins}
    out[mname] = res
print("Per-GT recall (greedy one-to-one match, top-100 dets) by the GT's max IoU with another GT of its class")
for key in ((0.001, 0.5), (0.25, 0.5), (0.001, 0.75)):
    print(f"\n score>={key[0]} IoU>={key[1]}")
    print("bin".ljust(14), "n_gt".rjust(7), *[m.rjust(15) for m in models])
    for _, _, lab in bins:
        n = out["yolo26m(o2o)"][key][lab][1]
        print(lab.ljust(14), str(n).rjust(7), *[f"{out[m][key][lab][0]:.4f}".rjust(15) for m in models])

# router signals
ids = np.array(imgs)
n = np.array([stats[i]["n"] for i in imgs]); p3 = np.array([stats[i]["p3"] for i in imgs]); p5 = np.array([stats[i]["p5"] for i in imgs])
ns = np.array([stats[i]["ns"] for i in imgs])
print("\nGT crowding: images with >=1 same-class pair IoU>0.3:", int((p3 > 0).sum()), " >0.5:", int((p5 > 0).sum()),
      " GT objects in pairs >0.5 occlusion share:", round(np.mean([gt_occ[k] > 0.5 for k in gt_occ]), 4))
m26 = dets_cache["yolo26m(o2o)"]
c25 = np.array([sum(1 for r in m26[i] if r[1] >= 0.25) for i in imgs]); c50 = np.array([sum(1 for r in m26[i] if r[1] >= 0.5) for i in imgs])
# o2o self-overlap: same-class pairs among dets with score>=0.25 with IoU>0.5 (crowd seen by the head itself)
def selfpairs(i, s=0.25, t=0.5):
    dd = [r for r in m26[i] if r[1] >= s]; tot = 0
    for c in set(r[0] for r in dd):
        b = [r[2:] for r in dd if r[0] == c]
        if len(b) > 1:
            M = iou_mat(b, b); tot += int((M[np.triu_indices(len(b), 1)] > t).sum())
    return tot
sp = np.array([selfpairs(i) for i in imgs])
for nm, sig in (("o2o count>=0.25", c25), ("o2o count>=0.5", c50), ("o2o self-pairs IoU>0.5", sp)):
    print(f"{nm:26s} Spearman vs GT count {spearmanr(sig, n)[0]:.3f}, vs GT pairs>0.3 {spearmanr(sig, p3)[0]:.3f}, vs GT pairs>0.5 {spearmanr(sig, p5)[0]:.3f}")
z = np.load(D + "val2017_stem_pooled.npz"); order = {v: k for k, v in enumerate(z["image_id"])}
rng = np.random.default_rng(0); fold = rng.integers(0, 5, len(imgs)); preds = {}
for feat in ("n320", "m640"):
    X = z[feat][[order[i] for i in imgs]]; X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = np.c_[X, np.ones(len(X))]
    for tname, y in (("log1p GT count", np.log1p(n)), ("log1p GT pairs>0.3", np.log1p(p3))):
        pred = np.zeros(len(y))
        for f in range(5):
            tr = fold != f
            A = X[tr]; w = np.linalg.solve(A.T @ A + 10.0 * np.eye(A.shape[1]), A.T @ y[tr]); pred[~tr] = X[~tr] @ w
        preds[(feat, tname)] = pred.copy()
        print(f"ridge {feat} -> {tname}: OOF Spearman vs GT count {spearmanr(pred, n)[0]:.3f}, vs GT pairs>0.3 {spearmanr(pred, p3)[0]:.3f}")
pickle.dump(dict(imgs=imgs, n=n, ns=ns, p3=p3, p5=p5, c25=c25, c50=c50, sp=sp, preds=preds), open(W + "crowd_stats.pkl", "wb"))
