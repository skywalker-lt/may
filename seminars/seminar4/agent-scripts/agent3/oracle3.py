#!/usr/bin/env python3
"""Agent 3, round 2: resolution oracle (C) on the new 512/768 dumps, shared-stem M/L oracle at the measured
branch costs (4.85 / 6.06 ms), random-route nulls, and a proxy-convention check (my TP/(TP+FP+FN) proxy versus a
per-image 101-point AP proxy as agent 5 used) on the same five dumps. Every mixture is scored globally with
pycocotools on full val2017. Exact Lagrangian choice: argmax_k q_k[i] - lambda L_k, lambda bisected to the budget."""
import json, os, sys, numpy as np, contextlib, io, gc
os.environ["OMP_NUM_THREADS"] = "2"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"; OUT = "/data/tmp/ds-yolo/seminar4/work/agent3"
LAT = {"n": 1.65, "s": 2.75, "m": 5.355, "l": 6.89, "x": 12.41, "m512": 3.781, "m768": 6.959, "s768": 3.617,
       "mS": 4.85, "lS": 6.06}  # mS/lS: M tail / L tail behind the shared M stem (moderator insert section 3)
FILES = {k: f"{D}/dump_yolo26{k}_coco.json" for k in "nslx"}
FILES["m"] = f"{D}/dumpml_yolo26m_coco.json"
FILES["m512"] = f"{D}/dumpml_yolo26m_512_coco.json"; FILES["m768"] = f"{D}/dumpml_yolo26m_768_coco.json"
FILES["s768"] = f"{D}/dumpml_yolo26s_768_coco.json"
TAU = 0.25
gt = COCO(f"{D}/instances_val2017.json"); img_ids = sorted(gt.getImgIds()); nI = len(img_ids)
rng = np.random.default_rng(0)


def run_eval(dets):
    dt = gt.loadRes(dets); ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = img_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev


def per_image(ev):
    """Returns (q_mine, q_ap, ngt, nsmall): my proxy, a per-image 101-point AP proxy (categories pooled), GT counts."""
    T = len(ev.params.iouThrs); K = len(ev.params.catIds); A = len(ev.params.areaRng)
    tp = np.zeros((T, nI)); fp = np.zeros((T, nI)); ng = np.zeros(nI); nsm = np.zeros(nI); qap = np.zeros(nI)
    sc = [[] for _ in range(nI)]; mt = [[] for _ in range(nI)]
    for k in range(K):
        for i in range(nI):
            e = ev.evalImgs[k * A * nI + i]
            if e is None: continue
            gi = np.array(e["gtIgnore"], dtype=bool); ng[i] += (~gi).sum()
            ds = np.array(e["dtScores"]); dm = np.array(e["dtMatches"]); di = np.array(e["dtIgnore"]).astype(bool)
            if len(ds):
                sc[i].append(ds); mt[i].append(((dm > 0) & ~di).astype(np.int8) - di.astype(np.int8))  # 1 tp, 0 fp, -1 ign
                keep = ds >= TAU
                if keep.any():
                    tp[:, i] += ((dm[:, keep] > 0) & ~di[:, keep]).sum(1); fp[:, i] += ((dm[:, keep] == 0) & ~di[:, keep]).sum(1)
    for a in gt.loadAnns(gt.getAnnIds(imgIds=img_ids)):
        if not a.get("iscrowd", 0) and a["area"] < 32 ** 2: nsm[img_ids.index(a["image_id"]) if False else idx[a["image_id"]]] += 1
    fn = ng[None, :] - tp; den = tp + fp + fn
    qm = np.where(den > 0, tp / np.maximum(den, 1), 1.0).mean(0)
    rt = np.linspace(0, 1, 101)
    for i in range(nI):
        if ng[i] == 0 or not sc[i]: qap[i] = 0.0 if ng[i] else 1.0; continue
        s = np.concatenate(sc[i]); m = np.concatenate(mt[i], axis=1); o = np.argsort(-s); m = m[:, o]
        aps = []
        for t in range(T):
            r = m[t]; keep = r >= 0; tpc = np.cumsum(r[keep] == 1); fpc = np.cumsum(r[keep] == 0)
            if len(tpc) == 0: aps.append(0.0); continue
            rec = tpc / ng[i]; prec = tpc / np.maximum(tpc + fpc, 1)
            for j in range(len(prec) - 2, -1, -1): prec[j] = max(prec[j], prec[j + 1])
            inds = np.searchsorted(rec, rt, side="left"); aps.append(np.mean([prec[j] if j < len(prec) else 0 for j in inds]))
        qap[i] = np.mean(aps)
    return qm, qap, ng, nsm


idx = {i: k for k, i in enumerate(img_ids)}
want = sys.argv[1].split(",") if len(sys.argv) > 1 else ["n", "s", "m", "l", "x", "m512", "m768", "s768"]
by_img, Q, QAP, AP = {}, {}, {}, {}
for k in want:
    dets = json.load(open(FILES[k])); ev = run_eval(dets); AP[k] = ev.stats[0]
    Q[k], QAP[k], NG, NSM = per_image(ev); del ev
    by_img[k] = {}
    for d in dets: by_img[k].setdefault(d["image_id"], []).append(d)
    del dets; gc.collect(); print(f"single {k}: AP {AP[k]:.4f}", flush=True)
for k2 in ("mS", "lS"):
    src = k2[0]
    if src in by_img: by_img[k2] = by_img[src]; Q[k2] = Q[src]; QAP[k2] = QAP[src]


def score(choice):
    return run_eval([d for i, k in zip(img_ids, choice) for d in by_img[k].get(i, [])]).stats[0]


def lagr(models, budget, q):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        lam = (lo + hi) / 2
        ch = [max(models, key=lambda k: q[k][i] - lam * LAT[k]) for i in range(nI)]
        if np.mean([LAT[k] for k in ch]) > budget: lo = lam
        else: hi = lam
    ch = [max(models, key=lambda k: q[k][i] - hi * LAT[k]) for i in range(nI)]
    return ch, float(np.mean([LAT[k] for k in ch]))


def shares(ch, models): return {k: round(ch.count(k) / nI, 3) for k in models}


def random_at(ch, models):
    return list(rng.permutation(ch))


def signal_route(sig, ch, models):
    """Give the heaviest models to the images ranked highest by sig, at the oracle's shares."""
    order = np.argsort(-sig); out = [None] * nI; pos = 0
    for k in sorted(models, key=lambda k: -LAT[k]):
        n = ch.count(k)
        for j in order[pos:pos + n]: out[j] = k
        pos += n
    return out


rows = []
def row(name, ch, models, avg=None):
    avg = avg if avg is not None else float(np.mean([LAT[k] for k in ch])); a = score(ch)
    rows.append((name, avg, a, shares(ch, models))); print(f"{name:52s} avg {avg:.3f} ms  AP {a:.4f}  {shares(ch, models)}", flush=True)
    return ch

for name, models, budget in [("C m512/m640/m768 @5.355", ["m512", "m", "m768"], 5.355),
                             ("C m512/m768 @5.355", ["m512", "m768"], 5.355),
                             ("C s768/m640/m768 @5.355", ["s768", "m", "m768"], 5.355),
                             ("C s768/m512/m640/m768 @5.355", ["s768", "m512", "m", "m768"], 5.355),
                             ("shared-stem M/L (4.85/6.06) @5.355", ["mS", "lS"], 5.355),
                             ("shared-stem M/L @5.355, 101pt proxy", ["mS", "lS"], 5.355),
                             ("five scales @5.36, my proxy", ["n", "s", "m", "l", "x"], 5.36),
                             ("five scales @5.36, 101pt AP proxy", ["n", "s", "m", "l", "x"], 5.36)]:
    if any(k not in by_img for k in models): continue
    q = QAP if "101pt" in name else Q
    ch, avg = lagr(models, budget, q); row(name + " ORACLE", ch, models, avg)
    row(name + " random null at oracle shares", random_at(ch, models), models, avg)
    if name.startswith("C") or name.startswith("shared"):
        row(name + " GT small-object count -> heavy", signal_route(NSM, ch, models), models, avg)
        row(name + " GT object count -> heavy", signal_route(NG, ch, models), models, avg)
json.dump({"single": AP, "rows": rows}, open(f"{OUT}/oracle3.json", "w"), indent=1)
