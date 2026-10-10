# CPU PTQ receipt on a 500-image val2017 subset: delta_all, delta_tail, the saturation signal, and precision mixtures.
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, pickle, numpy as np, io, contextlib
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent10"); from common import gt, per_image_ap, by_image
from pycocotools.cocoeval import COCOeval
from scipy.stats import spearmanr
R = "/data/tmp/ds-yolo/seminar5/work/agent10/r2"; G = gt()
ids = json.load(open(f"{R}/ids_fp32.json")); idset = set(ids)
def ev(dets, imgs=ids, pi=False):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(G, G.loadRes(dets) if dets else None, "bbox"); E.params.imgIds = list(imgs); E.evaluate(); E.accumulate(); E.summarize()
    return (E.stats[0], E.stats[3], E.stats[4], E.stats[5]) + ((per_image_ap(E),) if pi else ())
TA, TS, METH = (sys.argv[1], sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("q8all", "q8stem", "entropy")
D0 = {t: json.load(open(f"{R}/dets_{t}.json")) for t in ["fp32", TA, TS]}
D = {"fp32": D0["fp32"], "q8all": D0[TA], "q8stem": D0[TS]}
trt = [d for d in json.load(open("/data/tmp/ds-yolo/seminar5/inputs/dumps/dumpml_yolo26m_coco.json")) if d["image_id"] in idset]
r = {"trt_fp16_dump": ev(trt)}
for t, d in D.items(): r[t] = ev(d, pi=True)
for k, v in r.items(): print("%-14s AP %.4f  S/M/L %.4f/%.4f/%.4f" % ((k,) + tuple(v[:4])))
if "q8all" in r: print("delta_all  = fp32 - q8all  = %+.4f" % (r["fp32"][0] - r["q8all"][0]))
if "q8stem" in r:
    print("delta_tail = q8stem - q8all = %+.4f;  delta_stem = fp32 - q8stem = %+.4f" % (r["q8stem"][0] - r["q8all"][0], r["fp32"][0] - r["q8stem"][0]))
rng = np.random.default_rng(0); hs = []
pairs = [("fp32", "q8all")] + ([("q8stem", "q8all")] if "q8stem" in D else [])
for a, b in pairs:
    v = []
    for j in range(12):
        h = list(rng.permutation(ids)[:len(ids) // 2]); v.append(ev(D[a], h)[0] - ev(D[b], h)[0])
    print("split-half sd of %s - %s: %.4f -> se(500) est. %.4f" % (a, b, np.std(v), np.std(v) / np.sqrt(2)))
# saturation taps from the fp32 run
T = np.load(f"{R}/taps_fp32.npz"); names = list(T["names"]); tid = list(T["ids"]); assert tid == ids
frac, maxr = T["frac"], T["maxr"]
AJ = json.load(open(f"{R}/amax.json"))
if METH != "entropy": maxr = maxr * np.array([AJ[n]["entropy"] / AJ[n][METH] for n in T["names"]])[None]; frac = (maxr > 1).astype(float)  # frac becomes: tensor exceeded its calibrated range
import re
lay = np.array([int(re.match(r"/model\.(\d+)/", n).group(1)) if re.match(r"/model\.(\d+)/", n) else -1 for n in names])
l5 = names.index("/model.5/act/Mul_output_0")
feat = {"L5 max/amax": maxr[:, l5], "L5 frac>amax": frac[:, l5], "stem mean frac (L1-5)": frac[:, (lay >= 1) & (lay <= 5)].mean(1),
        "all-tensor mean frac": frac.mean(1), "all-tensor max of max/amax": maxr.max(1), "tail mean frac (L6-23)": frac[:, lay >= 6].mean(1)}
F = pickle.load(open("/data/tmp/ds-yolo/seminar5/work/agent10/features.pkl", "rb")); pos = {int(i): k for k, i in enumerate(F["ids"])}
ix = np.array([pos[i] for i in ids]); feat["GT count"] = F["cnt"][ix]; feat["n320 predicted count"] = F["pred_cnt"][ix]
print("median fraction of elements above amax: L5 %.2e, all tensors %.2e; images with any L5 clipping: %.2f" % (
    np.median(frac[:, l5]), np.median(frac.mean(1)), np.mean(frac[:, l5] > 0)))
for a, b, lab in [("fp32", "q8all", "dAP_all (fp32 - INT8)")] + ([("q8stem", "q8all", "dAP_tail (fp16 tail - INT8 tail)")] if "q8stem" in D else []):
    pa, pb = r[a][4], r[b][4]; d = np.array([pa.get(i, np.nan) - pb.get(i, np.nan) for i in ids]); h = ~np.isnan(d)
    top = np.sort(d[h])[::-1]; print(f"{lab}: images {h.sum()}, mean {np.mean(d[h]):+.4f}, sd {np.std(d[h]):.4f}, share >0 {np.mean(d[h] > 0):.2f}, share ==0 {np.mean(d[h] == 0):.2f}, top 30% carry {top[:int(0.3 * h.sum())].sum() / d[h].sum():.2f} of the sum")
    for k, v in feat.items(): print("    Spearman with %-28s %+.3f" % (k, spearmanr(d[h], v[h])[0]))
    for k in ["L5 max/amax", "all-tensor mean frac"]:
        from numpy.linalg import lstsq
        X = np.c_[np.ones(h.sum()), np.log1p(feat["GT count"][h])]
        res = lambda y: y - X @ lstsq(X, y, rcond=None)[0]
        print("    partial Spearman (GT count removed) with %-22s %+.3f" % (k, spearmanr(res(d[h]), res(feat[k][h]))[0]))
if "q8stem" in D:
    bA, bB = by_image(D["q8all"]), by_image(D["q8stem"]); dt = np.array([r["q8stem"][4].get(i, 0) - r["q8all"][4].get(i, 0) for i in ids])
    dl = r["q8stem"][0] - r["q8all"][0]
    for s in (0.2, 0.3, 0.4):
        k = int(round(s * len(ids)))
        def top(v): m = np.zeros(len(ids), bool); m[np.argsort(-v, kind="mergesort")[:k]] = True; return m
        def mixed(m): return [x for i, img in enumerate(ids) for x in (bB if m[i] else bA).get(img, [])]
        nul = np.mean([ev(mixed(rng.permutation(top(np.arange(len(ids)) * 0.0 - np.arange(len(ids))))))[0] for _ in range(5)])
        out = {"null (5 draws)": nul}
        for name, v in [("L5 max/amax", feat["L5 max/amax"]), ("L5 frac>amax", feat["L5 frac>amax"]), ("all-tensor mean frac (bound: not visible at L5)", feat["all-tensor mean frac"]),
                        ("n320 predicted count", feat["n320 predicted count"]), ("GT count", feat["GT count"]), ("per-image oracle", dt)]:
            out[name] = ev(mixed(top(v)))[0]
        print(f"share {s:.1f} to the fp16 tail: " + "; ".join("%s %.4f (g %+.4f, r %.2f)" % (n, a, a - nul, (a - nul) / ((1 - s) * dl) if dl > 0 else np.nan) for n, a in out.items()))
