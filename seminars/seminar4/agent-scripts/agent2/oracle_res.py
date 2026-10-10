"""Round 2: resolution-routing oracle (C) on the new 512/768 dumps, same proxy and rule as oracle.py.
Run: OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python oracle_res.py > oracle_res.log
"""
import json, os, sys, time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
OUT = "/data/tmp/ds-yolo/seminar4/work/agent2"
# T4 medians: moderator insert section 2 (512/768 rows), T4B for the rest
LAT = {"n": 1.65, "s": 2.75, "m": 5.36, "l": 6.89, "x": 12.41, "m512": 3.781, "m768": 6.959, "s768": 3.617}
FILES = {"n": "dump_yolo26n_coco.json", "s": "dump_yolo26s_coco.json", "m": "dumpml_yolo26m_coco.json",
         "l": "dump_yolo26l_coco.json", "x": "dump_yolo26x_coco.json",
         "m512": "dumpml_yolo26m_512_coco.json", "m768": "dumpml_yolo26m_768_coco.json",
         "s768": "dumpml_yolo26s_768_coco.json"}
gt = COCO(os.path.join(D, "instances_val2017.json"))
IMG = sorted(gt.getImgIds()); idx = {i: k for k, i in enumerate(IMG)}
rng = np.random.default_rng(0)

def score(res, tag):
    dt = gt.loadRes(res); ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = IMG
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"RESULT {tag}: AP={ev.stats[0]:.4f} AP50={ev.stats[1]:.4f} S/M/L={ev.stats[3]:.4f}/{ev.stats[4]:.4f}/{ev.stats[5]:.4f}", flush=True)
    return ev

cache = os.path.join(OUT, "proxy.npz"); q = {}
if os.path.exists(cache):
    z = np.load(cache); q = {k: z[k] for k in z.files}
dets = {k: json.load(open(os.path.join(D, f))) for k, f in FILES.items()}
# ground-truth per-image statistics for a physically visible route signal
n_small = np.zeros(len(IMG)); n_obj = np.zeros(len(IMG))
for a in gt.dataset["annotations"]:
    if a.get("iscrowd", 0): continue
    j = idx[a["image_id"]]; n_obj[j] += 1
    if a["area"] < 32 ** 2: n_small[j] += 1
for k in FILES:
    if k in q: continue
    t = time.time(); ev = score(dets[k], f"single {k}")
    qq = np.zeros(len(IMG)); fp = np.zeros(len(IMG)); ng = np.zeros(len(IMG)); tp = np.zeros((len(IMG), 10))
    for e in ev.evalImgs:
        if e is None or e["aRng"] != [0, 1e10] or e["maxDet"] != 100: continue
        j = idx[e["image_id"]]
        gi = np.array(e["gtIgnore"]).astype(bool); n_gt = int((~gi).sum()); ng[j] += n_gt
        if n_gt:
            tp[j] += (np.array(e["gtMatches"])[:, ~gi] > 0).sum(1)
        sc = np.array(e["dtScores"]); dm = np.array(e["dtMatches"]); di = np.array(e["dtIgnore"])
        if len(sc):
            fp[j] += int(((dm[0] == 0) & (~di[0].astype(bool)) & (sc > 0.3)).sum())
    has = ng > 0
    qq[has] = tp[has].mean(1) / ng[has] - 0.5 * fp[has] / ng[has]; qq[~has] = -0.1 * fp[~has]
    q[k] = qq; print(f"proxy {k}: mean={qq.mean():.4f} ({time.time()-t:.0f}s)", flush=True)
    np.savez(cache, **q)

by_img = {k: {} for k in FILES}
for k in FILES:
    for d in dets[k]: by_img[k].setdefault(d["image_id"], []).append(d)
def mixture(choice):
    res = []
    for j, i in enumerate(IMG): res += by_img[choice[j]].get(i, [])
    return res
def budget_choice(models, B):
    Q = np.stack([q[k] for k in models], 1); L = np.array([LAT[k] for k in models]); lo, hi = 0.0, 1.0
    for _ in range(60):
        lam = (lo + hi) / 2; c = (Q - lam * L).argmax(1)
        if L[c].mean() > B: lo = lam
        else: hi = lam
    return (Q - hi * L).argmax(1), L
def run_mix(models, B, tag, null=True, signal=True):
    c, L = budget_choice(models, B); shares = {k: float((c == i).mean()) for i, k in enumerate(models)}
    print(f"MIX {tag}: budget={B} avg_lat={L[c].mean():.3f} shares={shares}", flush=True)
    out = {"oracle": score(mixture([models[i] for i in c]), f"mix {tag}").stats[0], "avg": float(L[c].mean()), "shares": shares}
    if null:  # random assignment at the same shares: the zero-information router
        cr = rng.permutation(c); out["null"] = score(mixture([models[i] for i in cr]), f"null {tag}").stats[0]
    if signal:  # GT small-object count ranks images: most small objects -> costliest model, at the oracle's shares
        order = np.argsort(-(n_small + 1e-3 * n_obj), kind="stable"); cs = np.empty_like(c)
        by_cost = sorted(range(len(models)), key=lambda i: -L[i]); start = 0
        for i in by_cost:
            cnt = int((c == i).sum()); cs[order[start:start + cnt]] = i; start += cnt
        out["small_count_rule"] = score(mixture([models[i] for i in cs]), f"smallcount {tag}").stats[0]
    return out
res = {}
res["m512_m_m768@5.36"] = run_mix(["m512", "m", "m768"], 5.36, "m512_m_m768@5.36")
res["m512_m768@5.36"] = run_mix(["m512", "m768"], 5.36, "m512_m768@5.36")
res["m512_m_m768_s768@5.36"] = run_mix(["m512", "m", "m768", "s768"], 5.36, "m512_m_m768_s768@5.36", null=False, signal=False)
res["m512_m@5.36"] = run_mix(["m512", "m"], 5.36, "m512_m@5.36", null=False, signal=False)  # budget not binding: pure best-of-two
res["m_m768@6.0"] = run_mix(["m", "m768"], 6.0, "m_m768@6.0", null=True, signal=True)
res["s768_m@3.617"] = run_mix(["s768", "s"], 3.617, "s768_s@3.617", null=False, signal=False)
json.dump(res, open(os.path.join(OUT, "oracle_res_results.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
