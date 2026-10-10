"""Per-image route mixtures from cached COCOeval evalImgs (mixture AP is exact: COCO's per-image evaluation is
independent across images). Proxy question: does an NMS head of near-equal quality gain on crowded images, and can
the o2o head's own output or the stem features say which images? CPU, one thread."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, pickle, copy, contextlib, io, numpy as np
from scipy.stats import spearmanr
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
W = "/data/tmp/ds-yolo/seminar5/work/agent8/"
D = "/data/tmp/ds-yolo/seminar5/inputs/dumps/"
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(D + "instances_val2017.json")
E = COCOeval(gt, None, "bbox"); E._paramsEval = copy.deepcopy(E.params)
P = E.params; I = len(P.imgIds); K = len(P.catIds); A = len(P.areaRng)
C = {m: pickle.load(open(W + f"cache/{m}.pkl", "rb")) for m in sys.argv[1].split(",")}
for m in C:
    assert C[m]["imgIds"] == P.imgIds
S = pickle.load(open(W + "crowd_stats.pkl", "rb")); assert S["imgs"] == P.imgIds
img_of = np.tile(np.arange(I), K * A)

def ap_of(evalImgs):
    E.evalImgs = evalImgs
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate(); E.summarize()
    return E.stats[0], E.stats[3], E.stats[4], E.stats[5]

def mix(route, a, b):  # route[i] True -> model b
    r = route[img_of]; ea, eb = C[a]["evalImgs"], C[b]["evalImgs"]
    return [y if t else x for x, y, t in zip(ea, eb, r)]

def subset(mask, a):
    r = mask[img_of]; ea = C[a]["evalImgs"]
    return [x if t else None for x, t in zip(ea, r)]

def per_image_ap(m):
    """COCO-style AP per image: area 'all', maxDet 100, mean over categories with GT in the image (101-pt)."""
    ev = C[m]["evalImgs"]; T = len(P.iouThrs); rec = np.linspace(0, 1, 101)
    out = np.full(I, np.nan); acc = [[] for _ in range(I)]
    for k in range(K):
        base = k * A * I  # area index 0 = all
        for i in range(I):
            e = ev[base + i]
            if e is None:
                continue
            gi = np.asarray(e["gtIgnore"]).astype(bool); npig = int((~gi).sum())
            if npig == 0:
                continue
            sc = np.asarray(e["dtScores"][:100]); o = np.argsort(-sc, kind="mergesort")
            dm = np.asarray(e["dtMatches"])[:, :100][:, o]; di = np.asarray(e["dtIgnore"])[:, :100][:, o]
            tps = np.logical_and(dm > 0, ~di); fps = np.logical_and(dm == 0, ~di)
            tp = np.cumsum(tps, 1).astype(float); fp = np.cumsum(fps, 1).astype(float)
            aps = []
            for t in range(T):
                if tp.shape[1] == 0:
                    aps.append(0.0); continue
                rc = tp[t] / npig; pr = tp[t] / (tp[t] + fp[t] + 1e-12)
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                ids = np.searchsorted(rc, rec, side="left")
                q = np.where(ids < len(pr), pr[np.minimum(ids, len(pr) - 1)], 0.0)
                aps.append(q.mean())
            acc[i].append(np.mean(aps))
    for i in range(I):
        if acc[i]:
            out[i] = np.mean(acc[i])
    return out

base, alt = sys.argv[2], sys.argv[3]
print(f"base {base} (o2o), alternative {alt}")
for m in (base, alt):
    print(f"{m:6s} full AP {ap_of(C[m]['evalImgs'])[0]:.4f}")
n, p3, p5, c25, sp = S["n"], S["p3"], S["p5"], S["c25"], S["sp"]
print("\nSubset AP (same images for both models)")
for lab, mask in (("no same-class pair IoU>0.3", p3 == 0), ("any pair IoU>0.3", p3 > 0), ("any pair IoU>0.5", p5 > 0),
                  ("GT count >= 10", n >= 10), ("GT count < 10", n < 10), ("o2o count>=0.25 at least 10", c25 >= 10)):
    a1 = ap_of(subset(mask, base))[0]; a2 = ap_of(subset(mask, alt))[0]
    print(f"{lab:32s} images {int(mask.sum()):5d}  {base} {a1:.4f}  {alt} {a2:.4f}  alt-base {a2 - a1:+.4f}")
pa, pb = per_image_ap(base), per_image_ap(alt); g = pb - pa; ok = ~np.isnan(g)
print(f"\nper-image AP gain alt-base: mean {np.nanmean(g):+.4f}, share of images where alt better {np.mean(g[ok] > 0):.3f}, worse {np.mean(g[ok] < 0):.3f}")
preds = S["preds"]
sigs = {"GT pairs>0.3": p3 + 1e-3 * n, "GT count": n.astype(float), "o2o count>=0.25": c25 + 1e-3 * sp, "o2o self-pairs>0.5": sp + 1e-3 * c25,
        "ridge m640->pairs": preds[("m640", "log1p GT pairs>0.3")], "ridge n320->count": preds[("n320", "log1p GT count")]}
for k, v in sigs.items():
    print(f"Spearman({k}, per-image gain) = {spearmanr(v[ok], g[ok])[0]:+.3f}")
rng = np.random.default_rng(1)
for share in (0.1, 0.2, 0.3):
    q = int(round(share * I)); rows = []
    gg = np.where(ok, g, -1e9); r = np.zeros(I, bool); r[np.argsort(-gg)[:q]] = True
    rows.append(("oracle (per-image gain)", ap_of(mix(r, base, alt))[0]))
    for k, v in sigs.items():
        r = np.zeros(I, bool); r[np.argsort(-v, kind="mergesort")[:q]] = True
        rows.append((f"rule: top {k}", ap_of(mix(r, base, alt))[0]))
    nulls = []
    for d in range(3):
        r = np.zeros(I, bool); r[rng.choice(I, q, replace=False)] = True; nulls.append(ap_of(mix(r, base, alt))[0])
    rows.append(("share null (mean of 3)", np.mean(nulls)))
    print(f"\nshare {share:.1f} ({q} images to {alt}); null draws {', '.join(f'{x:.4f}' for x in nulls)}")
    for lab, v in rows:
        print(f"  {lab:28s} {v:.4f}")
