# Exact per-image routing between two pycocotools variants by recombining evalImgs, then accumulate().
import pickle, copy, numpy as np, os, sys, contextlib, io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'; W = '/data/tmp/ds-yolo/seminar5/work/agent3/'
with contextlib.redirect_stdout(io.StringIO()): gt = COCO(D + 'instances_val2017.json')
E0 = COCOeval(gt, None, 'bbox'); I = len(E0.params.imgIds); A = len(E0.params.areaRng); K = len(E0.params.catIds)
imgpos = {im: j for j, im in enumerate(E0.params.imgIds)}
_cache = {}
def load(tag, v):
    if (tag, v) not in _cache: _cache[(tag, v)] = pickle.load(open(W + f'ei_{tag}_{v}.pkl', 'rb'))['evalImgs']
    return _cache[(tag, v)]
def score(base, alt, routed_ids):
    """base/alt: lists of evalImgs; routed_ids: image ids taking alt. Returns AP, APs, APm, APl."""
    mask = np.zeros(I, bool); mask[[imgpos[i] for i in routed_ids]] = True
    m = np.tile(mask, K * A)
    ev = [a if r else b for b, a, r in zip(base, alt, m)]
    E = COCOeval(gt, None, 'bbox'); E.params.imgIds = sorted(gt.getImgIds()); E.params.catIds = sorted(gt.getCatIds())
    E._paramsEval = copy.deepcopy(E.params); E.evalImgs = ev
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0], E.stats[3], E.stats[4], E.stats[5]
def score_multi(variants, assign):
    """variants: list of evalImgs lists; assign: array over E0.params.imgIds order (index into variants)."""
    m = np.tile(np.asarray(assign), K * A)
    ev = [variants[j][t] for t, j in enumerate(m)]
    E = COCOeval(gt, None, 'bbox'); E.params.imgIds = sorted(gt.getImgIds()); E.params.catIds = sorted(gt.getCatIds())
    E._paramsEval = copy.deepcopy(E.params); E.evalImgs = ev
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0], E.stats[3], E.stats[4], E.stats[5]
