"""Exact COCO AP of per-image mixtures: evalImgs entries are per (category, area, image), so a mixture's evalImgs is the
selected model's entry for each image; accumulate()+summarize() then give the exact pycocotools AP of the mixed dump."""
import os, copy, pickle, contextlib, io, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = "/data/tmp/ds-yolo/seminar5/inputs/dumps"; W = "/data/tmp/ds-yolo/seminar5/work/agent7/ev"
with contextlib.redirect_stdout(io.StringIO()):
    GT = COCO(f"{D}/instances_val2017.json")
_cache = {}
def load(nm):
    if nm not in _cache: _cache[nm] = pickle.load(open(f"{W}/{nm}.pkl", "rb"))
    return _cache[nm]
def template():
    E = COCOeval(GT, None, "bbox") if False else COCOeval.__new__(COCOeval)
    E.cocoGt = GT; E.cocoDt = None; E.params = __import__("pycocotools.cocoeval", fromlist=["Params"]).Params(iouType="bbox")
    E.params.imgIds = sorted(GT.getImgIds()); E.params.catIds = sorted(GT.getCatIds())
    E._paramsEval = copy.deepcopy(E.params); E.eval = {}; E.stats = []
    return E
_T = None
def mix_ap(names, assign, full=False):
    """names: list of dump names; assign: int array over the 5000 sorted image ids."""
    global _T
    if _T is None: _T = template()
    evs = [load(n)["evalImgs"] for n in names]; I = len(_T.params.imgIds); K = len(_T.params.catIds); A = 4
    assign = np.asarray(assign); out = [None] * (K * A * I)
    idx = np.arange(K * A * I); img = idx % I; src = assign[img]
    for j, ev in enumerate(evs):
        for p in idx[src == j]: out[p] = ev[p]
    _T.evalImgs = out
    with contextlib.redirect_stdout(io.StringIO()):
        _T.accumulate(); _T.summarize()
    return _T.stats.copy() if full else _T.stats[0]
def img_ids(): return sorted(GT.getImgIds())
def gt_counts():
    ids = img_ids(); n = np.zeros(len(ids)); ns = np.zeros(len(ids)); area = np.zeros(len(ids))
    for i, iid in enumerate(ids):
        anns = [a for a in GT.loadAnns(GT.getAnnIds(imgIds=iid, iscrowd=False))]
        n[i] = len(anns); ns[i] = sum(a["area"] < 32 ** 2 for a in anns); area[i] = np.median([a["area"] for a in anns]) if anns else 0
    return n, ns, area
def mix_ap_subset(names, assign, mask):
    """Exact AP of the mixture restricted to the images where mask is True (e.g. a split half of val2017)."""
    E = template(); allids = E.params.imgIds; I = len(allids); sel = np.where(mask)[0]; Is = len(sel)
    E.params.imgIds = [allids[i] for i in sel]; E._paramsEval = copy.deepcopy(E.params)
    evs = [load(n)["evalImgs"] for n in names]; K = len(E.params.catIds); A = 4; out = [None] * (K * A * Is)
    for k in range(K):
        for a in range(A):
            base, bs = k * A * I + a * I, k * A * Is + a * Is
            for j, i in enumerate(sel): out[bs + j] = evs[assign[i]][base + i]
    E.evalImgs = out
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate(); E.summarize()
    return E.stats[0]
def mix_ap_index(names, assign, sel):
    """Exact AP of the mixture over the image index list sel (duplicates allowed: bootstrap resamples)."""
    E = template(); allids = E.params.imgIds; I = len(allids); sel = np.asarray(sel); Is = len(sel)
    E.params.imgIds = list(range(Is)); E._paramsEval = copy.deepcopy(E.params)
    evs = [load(n)["evalImgs"] for n in names]; K = len(E.params.catIds); A = 4; out = [None] * (K * A * Is)
    for k in range(K):
        for a in range(A):
            base, bs = k * A * I + a * I, k * A * Is + a * Is
            for j, i in enumerate(sel): out[bs + j] = evs[assign[i]][base + i]
    E.evalImgs = out
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate(); E.summarize()
    return E.stats[0]
