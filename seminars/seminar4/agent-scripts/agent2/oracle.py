#!/usr/bin/env python3
"""Oracle headroom for input-adaptive designs from the per-image T4 dumps.

Step 1: evaluate every dump once with pycocotools; from evalImgs build a per-image proxy
        q[model][img] = mean over the 10 IoU thresholds of (matched GT / GT) - 0.5*FP(score>0.3)/max(GT,1)
        (area=all, maxDet=100).  Images with no GT get proxy = -FP(score>0.3)*0.1.
Step 2: for a latency budget B (average ms per image), choose per image the model maximising
        q - lam*lat, binary-search lam so that the average latency == B; then score the mixture json globally.
Step 3: cascade oracle: cheap model always, heavy model on the best fraction p of images (by proxy gain).
Step 4: cheap-confidence routing (non-oracle): heavy model on the images whose cheap-model max score is lowest.
"""
import json, sys, os, time
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
OUT = "/data/tmp/ds-yolo/seminar4/work/agent2"
LAT = {"n": 1.65, "s": 2.75, "m": 5.36, "l": 6.89, "x": 12.41, "v12m": 5.53, "11m": 5.24}
FILES = {"n": "dump_yolo26n_coco.json", "s": "dump_yolo26s_coco.json", "m": "dumpml_yolo26m_coco.json",
         "l": "dump_yolo26l_coco.json", "x": "dump_yolo26x_coco.json", "v12m": "dump_yolov12m_sdpa_coco.json",
         "11m": "dump_yolo11m_coco.json"}
gt = COCO(os.path.join(D, "instances_val2017.json"))
IMG = sorted(gt.getImgIds())
idx = {i: k for k, i in enumerate(IMG)}

def score(res, tag):
    dt = gt.loadRes(res)
    ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = IMG
    ev.evaluate(); ev.accumulate(); ev.summarize()
    print(f"RESULT {tag}: AP={ev.stats[0]:.4f} AP50={ev.stats[1]:.4f} S/M/L={ev.stats[3]:.4f}/{ev.stats[4]:.4f}/{ev.stats[5]:.4f}", flush=True)
    return ev

cache = os.path.join(OUT, "proxy.npz")
dets, q = {}, {}
for k, f in FILES.items():
    dets[k] = json.load(open(os.path.join(D, f)))
if os.path.exists(cache):
    z = np.load(cache); q = {k: z[k] for k in z.files}
for k in FILES:
    if k in q: continue
    if True:
        t = time.time(); ev = score(dets[k], f"single {k}")
        qq = np.zeros(len(IMG)); fp = np.zeros(len(IMG)); ng = np.zeros(len(IMG)); tp = np.zeros((len(IMG), 10))
        for e in ev.evalImgs:
            if e is None or e["aRng"] != [0, 1e10] or e["maxDet"] != 100: continue
            j = idx[e["image_id"]]
            gi = np.array(e["gtIgnore"]).astype(bool); n_gt = int((~gi).sum()); ng[j] += n_gt
            if n_gt:
                m = np.array(e["gtMatches"])[:, ~gi] > 0  # 10 x n_gt
                tp[j] += m.sum(1)
            sc = np.array(e["dtScores"]); dm = np.array(e["dtMatches"]); di = np.array(e["dtIgnore"])
            if len(sc):
                # FP at IoU=0.5 with score>0.3, not ignored
                fp[j] += int(((dm[0] == 0) & (~di[0].astype(bool)) & (sc > 0.3)).sum())
        has = ng > 0
        qq[has] = tp[has].mean(1) / ng[has] - 0.5 * fp[has] / ng[has]
        qq[~has] = -0.1 * fp[~has]
        q[k] = qq; print(f"proxy {k}: mean={qq.mean():.4f} ({time.time()-t:.0f}s)", flush=True)
        np.savez(cache, **q)

by_img = {k: {} for k in FILES}
for k in FILES:
    for d in dets[k]: by_img[k].setdefault(d["image_id"], []).append(d)

def mixture(choice):  # choice: array of model keys per image index
    res = []
    for j, i in enumerate(IMG): res += by_img[choice[j]].get(i, [])
    return res

def budget_mix(models, B, tag):
    Q = np.stack([q[k] for k in models], 1); L = np.array([LAT[k] for k in models])
    lo, hi = 0.0, 1.0
    for _ in range(60):
        lam = (lo + hi) / 2; c = (Q - lam * L).argmax(1)
        if L[c].mean() > B: lo = lam
        else: hi = lam
    lam = hi; c = (Q - lam * L).argmax(1)
    shares = {k: float((c == i).mean()) for i, k in enumerate(models)}
    print(f"MIX {tag}: budget={B} avg_lat={L[c].mean():.3f} lam={lam:.4f} shares={shares}", flush=True)
    ev = score(mixture([models[i] for i in c]), f"mix {tag}"); return ev.stats[0], L[c].mean(), shares

def cascade(cheap, heavy, B, tag, by="oracle"):
    L_heavy = LAT[heavy]; p = min(1.0, (B - LAT[cheap]) / L_heavy)
    n_heavy = int(round(p * len(IMG)))
    if by == "oracle": key = q[heavy] - q[cheap]
    else:  # lowest cheap max-confidence first
        mx = np.zeros(len(IMG))
        for j, i in enumerate(IMG):
            s = [d["score"] for d in by_img[cheap].get(i, [])]; mx[j] = max(s) if s else 0.0
        key = -mx
    order = np.argsort(-key); c = np.array([cheap] * len(IMG), dtype=object); c[order[:n_heavy]] = heavy
    avg = LAT[cheap] + p * L_heavy
    print(f"CASCADE {tag} ({by}): cheap={cheap} heavy={heavy} p_heavy={p:.3f} avg_lat={avg:.3f}", flush=True)
    ev = score(mixture(list(c)), f"cascade {tag} {by}"); return ev.stats[0], avg

out = {}
if "--mix" in sys.argv:
    out["oracle_full_nsmlx@5.36"] = budget_mix(["n", "s", "m", "l", "x"], 5.36, "nsmlx@5.36")
    out["oracle_nml@5.36"] = budget_mix(["n", "m", "l"], 5.36, "nml@5.36")
    out["oracle_sml@5.36"] = budget_mix(["s", "m", "l"], 5.36, "sml@5.36")
    out["oracle_ml@5.36"] = budget_mix(["m", "l"], 5.36, "ml@5.36")
    out["oracle_sl@5.36"] = budget_mix(["s", "l"], 5.36, "sl@5.36")
    out["oracle_m_v12m_11m@5.36"] = budget_mix(["m", "v12m", "11m"], 5.36, "m_v12m_11m@5.36")
    out["oracle_m_v12m_11m@5.53"] = budget_mix(["m", "v12m", "11m"], 5.53, "m_v12m_11m@5.53")
    out["oracle_nsmlx@6.89"] = budget_mix(["n", "s", "m", "l", "x"], 6.89, "nsmlx@6.89")
    out["oracle_nsmlx@unlimited"] = budget_mix(["n", "s", "m", "l", "x"], 99, "nsmlx@unl")
if "--cascade" in sys.argv:
    out["cascade_n_l_oracle"] = cascade("n", "l", 5.36, "n->l@5.36", "oracle")
    out["cascade_n_l_conf"] = cascade("n", "l", 5.36, "n->l@5.36", "conf")
    out["cascade_n_m_oracle"] = cascade("n", "m", 5.36, "n->m@5.36", "oracle")
    out["cascade_n_m_conf"] = cascade("n", "m", 5.36, "n->m@5.36", "conf")
    out["cascade_s_l_oracle"] = cascade("s", "l", 5.36, "s->l@5.36", "oracle")
    out["cascade_m_l_full"] = cascade("m", "l", 12.25, "m->l all", "oracle")  # m plus l on every image: AP ceiling of union
if "--null" in sys.argv:  # random partition at matched shares, the null for the m/l oracle
    rng = np.random.default_rng(0)
    for s_l in (0.25, 0.5):
        c = np.where(rng.random(len(IMG)) < s_l, "l", "m")
        ev = score(mixture(list(c)), f"random m/l share_l={s_l}")
        out[f"random_ml_{s_l}"] = ev.stats[0]
json.dump(out, open(os.path.join(OUT, "oracle_results_" + "_".join(a.strip('-') for a in sys.argv[1:]) + ".json"), "w"), indent=1)
print(json.dumps(out, indent=1))
