"""Exact per-image mixing of COCO dumps: evaluate each dump once, then any per-image route is scored by
assembling evalImgs from the chosen source per image and running accumulate() (identical to scoring the mixed dump)."""
import os, json, pickle, numpy as np, contextlib, io
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = "/data/tmp/ds-yolo/seminar5/inputs/dumps"
CACHE = os.environ.get("MIXCACHE", "/tmp/claude-0/-/41c4b87a-be09-4002-ae56-c31dc4dde162/scratchpad/mixcache")
os.makedirs(CACHE, exist_ok=True)
_gt = None
def gt():
    global _gt
    if _gt is None:
        with contextlib.redirect_stdout(io.StringIO()): _gt = COCO(f"{D}/instances_val2017.json")
    return _gt
def evaluated(name):
    pk = f"{CACHE}/{name}.pkl"
    if os.path.exists(pk):
        return pickle.load(open(pk, "rb"))
    G = gt(); dets = json.load(open(f"{D}/{name}.json"))
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
            dt = G.loadRes(json.load(open(f"{D}/dumpml_yolo26m_coco.json"))[:10]); E = COCOeval(G, dt, "bbox")
        _base = E
    return _base
def score(sources, choice):
    """sources: list of evaluated() dicts; choice: int array over imgIds order (index into sources). Returns AP, APs, APm, APl."""
    E = base_eval(); S = np.stack([s["ev"] for s in sources])  # S,C,A,I
    I = S.shape[3]; ch = np.asarray(choice).astype(int)
    mixed = S[ch, :, :, np.arange(I)]  # I,C,A
    mixed = np.transpose(mixed, (1, 2, 0)).reshape(-1)
    E.evalImgs = list(mixed); E.params.imgIds = sources[0]["imgIds"]; E._paramsEval = E.params
    import copy
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate(); E.summarize()
    st = E.stats
    return round(float(st[0]), 4), round(float(st[3]), 4), round(float(st[4]), 4), round(float(st[5]), 4)
