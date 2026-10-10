"""Per-image mixing of cached evalImgs (exact multi-label pycocotools on full val2017) plus the est. T4 rect cost model.
Policies: fixed-long-side rect, pixel-cap rect, routed rect ladders (GT-count rule and out-of-fold ridge on the
pooled YOLO26-N@320 stem feature), and the square/rect envelopes.  CPU, one thread."""
import json, pickle, sys, time, os, copy
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cost import cost_interp, cost_linfit, rect_shape, kpix

IN = '/data/tmp/ds-yolo/seminar6/inputs'
CACHE = '/data/tmp/ds-yolo/seminar6/work/agent5/cache'
ROUTER_MS = 0.18   # measured stem-only N@320 router on the T4 (Phase 2); charged on every image when a count router is used

gt = COCO(f'{IN}/dumps/instances_val2017.json')
imgIds = sorted(gt.getImgIds())
I = len(imgIds); pos = {iid: n for n, iid in enumerate(imgIds)}
imginfo = {im['id']: (im['width'], im['height']) for im in gt.loadImgs(imgIds)}
W = np.array([imginfo[i][0] for i in imgIds]); H = np.array([imginfo[i][1] for i in imgIds])

# GT counts per image
cnt_all = np.zeros(I); cnt_sm = np.zeros(I)
for a in gt.loadAnns(gt.getAnnIds(iscrowd=False)):
    n = pos[a['image_id']]; cnt_all[n] += 1
    if a['area'] < 96 ** 2: cnt_sm[n] += 1

_cache = {}
def load(name):
    if name not in _cache:
        with open(f'{CACHE}/{name}.pkl', 'rb') as f:
            _cache[name] = pickle.load(f)
    return _cache[name]

_E = None
def evaluator():
    global _E
    if _E is None:
        dt = gt.loadRes(f'{IN}/dumps/dumpml_yolo26m_coco.json')
        _E = COCOeval(gt, dt, 'bbox'); _E.params.imgIds = imgIds
    return _E

def mix_ap(assign):
    """assign: array of dump names per image position. Returns stats (AP, AP50, AP75, S, M, L)."""
    names = sorted(set(assign))
    ref = load(names[0]); p = ref['params']
    K = len(p.catIds); A = len(p.areaRng); assert len(p.imgIds) == I
    ev = [None] * (K * A * I)
    masks = {nm: np.where(np.array(assign) == nm)[0] for nm in names}
    for nm in names:
        src = load(nm)['evalImgs']; idx = masks[nm]
        for k in range(K):
            for a in range(A):
                base = k * A * I + a * I
                for i in idx:
                    ev[base + i] = src[base + i]
    E = evaluator(); E.evalImgs = ev; E.params = copy.deepcopy(p); E._paramsEval = copy.deepcopy(p)
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate(); E.summarize()
    return E.stats[[0, 1, 2, 3, 4, 5]].copy()

def dump_stats(name):
    return np.array(load(name)['stats'])[[0, 1, 2, 3, 4, 5]]

def rect_cost(model, s, n, fn=cost_interp):
    w, h = rect_shape(s, W[n], H[n]); return fn(model, w, h)

def policy_cost(model, scales, fn=cost_interp):
    c = np.array([rect_cost(model, scales[n], n, fn) for n in range(I)]); return c.mean(), c.max()

SQ_PTS = {'l': [(4.28, 0.5117), (4.88, 0.5262), (5.32, 0.5329), (6.25, 0.5368), (6.89, 0.5417)],
          'm': [(3.33, 0.4937), (3.78, 0.5063), (4.93, 0.5203), (5.20, 0.5234), (5.34, 0.5261)]}
ALL_SQ = sorted(SQ_PTS['l'] + SQ_PTS['m'])

def hull(points):
    pts = sorted(points); h = []
    for p in pts:
        while len(h) >= 2:
            (x1, y1), (x2, y2) = h[-2], h[-1]
            if (y2 - y1) * (p[0] - x1) <= (p[1] - y1) * (x2 - x1): h.pop()
            else: break
        h.append(p)
    return h

def env_at(points, t):
    h = hull(points); xs = [p[0] for p in h]; ys = [p[1] for p in h]
    if t <= xs[0]: return ys[0]
    if t >= xs[-1]: return ys[-1]
    return float(np.interp(t, xs, ys))

def fmt(st): return f'AP {st[0]:.4f} (50 {st[1]:.4f} 75 {st[2]:.4f} S/M/L {st[3]:.4f}/{st[4]:.4f}/{st[5]:.4f})'
