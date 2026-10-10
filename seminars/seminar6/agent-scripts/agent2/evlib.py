"""Exact per-image mixing of COCO dumps (evalImgs assembly, same method as seminar 5's mixlib), pointed at seminar6 inputs."""
import os, json, pickle, numpy as np, contextlib, io
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
INP = "/data/tmp/ds-yolo/seminar6/inputs"
CACHE = "/data/tmp/ds-yolo/seminar6/work/agent2/cache"; os.makedirs(CACHE, exist_ok=True)
PATHS = {"l448": f"{INP}/dumps_r2/dumpml_yolo26l_448_coco.json", "l512": f"{INP}/dumps_r2/dumpml_yolo26l_512_coco.json",
         "l544": f"{INP}/dumps_r2/dumpml_yolo26l_544_coco.json", "l576": f"{INP}/dumps_r2/dumpml_yolo26l_576_coco.json",
         "l640": f"{INP}/dumps/dump_yolo26l_coco.json", "m640": f"{INP}/dumps/dumpml_yolo26m_coco.json",
         "m448": f"{INP}/dumps_r2/dumpml_yolo26m_448_coco.json", "n640": f"{INP}/dumps/dump_yolo26n_coco.json",
         "s640": f"{INP}/dumps/dump_yolo26s_coco.json", "x640": f"{INP}/dumps/dump_yolo26x_coco.json"}
_gt = None
def gt(path=f"{INP}/dumps/instances_val2017.json"):
    global _gt
    if _gt is None:
        with contextlib.redirect_stdout(io.StringIO()): _gt = COCO(path)
    return _gt
def full_eval(dets, G=None):
    G = G or gt()
    with contextlib.redirect_stdout(io.StringIO()):
        dt = G.loadRes(dets); E = COCOeval(G, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    return E
def evaluated(name):
    pk = f"{CACHE}/{name}.pkl"
    if os.path.exists(pk): return pickle.load(open(pk, "rb"))
    G = gt(); dets = json.load(open(PATHS[name]))
    with contextlib.redirect_stdout(io.StringIO()):
        dt = G.loadRes(dets); E = COCOeval(G, dt, "bbox"); E.evaluate()
    p = E.params; C, A, I = len(p.catIds), len(p.areaRng), len(p.imgIds)
    arr = np.empty(len(E.evalImgs), dtype=object); arr[:] = E.evalImgs
    out = dict(ev=arr.reshape(C, A, I), imgIds=list(p.imgIds))
    pickle.dump(out, open(pk, "wb"), protocol=4); return out
_base = None
def base_eval():
    global _base
    if _base is None:
        G = gt()
        with contextlib.redirect_stdout(io.StringIO()):
            dt = G.loadRes(json.load(open(PATHS["m640"]))[:10]); _base = COCOeval(G, dt, "bbox")
    return _base
def score(sources, choice):
    E = base_eval(); S = np.stack([s["ev"] for s in sources]); I = S.shape[3]; ch = np.asarray(choice).astype(int)
    mixed = np.transpose(S[ch, :, :, np.arange(I)], (1, 2, 0)).reshape(-1)
    E.evalImgs = list(mixed); E.params.imgIds = sources[0]["imgIds"]; E._paramsEval = E.params
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    st = E.stats; return round(float(st[0]), 4), round(float(st[3]), 4), round(float(st[4]), 4), round(float(st[5]), 4)
def per_image_proxy(src, thr=0.3):
    """Per-image proxy score: mean over IoU thresholds of (TP - FP) among dets with score>thr, over all cats, area=all."""
    ev = src["ev"][:, 0, :]; C, I = ev.shape; out = np.zeros(I)
    for i in range(I):
        tp = 0.0; fp = 0.0
        for c in range(C):
            e = ev[c, i]
            if e is None: continue
            sc = np.asarray(e["dtScores"]); keep = sc > thr
            if keep.sum() == 0: continue
            m = np.asarray(e["dtMatches"])[:, keep]; ig = np.asarray(e["dtIgnore"])[:, keep]
            tp += ((m > 0) & ~ig).sum() / m.shape[0]; fp += ((m == 0) & ~ig).sum() / m.shape[0]
        out[i] = tp - fp
    return out
def gt_stats():
    G = gt(); ids = sorted(G.getImgIds()); cnt = np.zeros(len(ids)); small = np.zeros(len(ids)); med = np.zeros(len(ids))
    for k, i in enumerate(ids):
        for a in G.loadAnns(G.getAnnIds(imgIds=i, iscrowd=False)):
            cnt[k] += 1; ar = a["area"]
            if ar < 32 ** 2: small[k] += 1
            elif ar < 96 ** 2: med[k] += 1
    return ids, cnt, small, med
