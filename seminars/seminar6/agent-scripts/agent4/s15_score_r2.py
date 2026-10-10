import sys, json, io, contextlib, numpy as np
sys.path.insert(0, '/data/tmp/ds-yolo/seminar6/work/agent4'); from fastap import ap_from
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from common import GT, OUT
with contextlib.redirect_stdout(io.StringIO()): gt = COCO(GT)
E = json.load(open(f'{OUT}/eval_ids.json')); sub = E['sub400'][:120]; n = len(sub); ii = {im: i for i, im in enumerate(sub)}
catIds = sorted(gt.getCatIds()); ci = {c: k for k, c in enumerate(catIds)}
R = {}
for f in ['matrix_dets.json', 'matrix2_dets.json', 'matrix3_dets.json']:
    for k, v in json.load(open(f'{OUT}/{f}')).items(): R.setdefault(k, []).extend(v)
for k, v in json.load(open(f'{OUT}/matrix6_dets.json')).items():
    if k != '_done': R.setdefault(k, []).extend(v)
keys = ['clean__public', 'clean__gn3_k21', 'clean__ct3_k21', 'clean__pooled_k21', 'gn3__public', 'gn3__gn3_k21', 'gn3__pooled_k21', 'ct3__public', 'ct3__ct3_k21', 'ct3__pooled_k21',
        'gn1__public', 'gn1__gn3_k21', 'gn5__public', 'gn5__gn3_k21', 'ct1__public', 'ct1__ct3_k21']
def rec_of(dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets); Ev = COCOeval(gt, dt, 'bbox'); Ev.params.imgIds = sub; Ev.params.areaRng = [[0, 1e10]]; Ev.params.areaRngLbl = ['all']; Ev.params.maxDets = [100]; Ev.evaluate()
    cats = []; imgs = []; sc = []; m = []; ig = []; npig = np.zeros((80, n), np.int32)
    for e in Ev.evalImgs:
        if e is None: continue
        k = ci[e['category_id']]; i = ii[e['image_id']]; npig[k, i] = int(np.sum(np.logical_not(e['gtIgnore'])))
        if len(e['dtScores']) == 0: continue
        cats.append(np.full(len(e['dtScores']), k)); imgs.append(np.full(len(e['dtScores']), i)); sc.append(np.array(e['dtScores'])); m.append(e['dtMatches'] > 0); ig.append(e['dtIgnore'].astype(bool))
    return dict(cat=np.concatenate(cats), img=np.concatenate(imgs), score=np.concatenate(sc), match=np.concatenate(m, 1), ign=np.concatenate(ig, 1), npig=npig)
def full_ap(dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets); Ev = COCOeval(gt, dt, 'bbox'); Ev.params.imgIds = sub; Ev.evaluate(); Ev.accumulate(); Ev.summarize()
    return Ev.stats[0]
def ap_assign(recs, assign, idx=None):
    if idx is None: idx = np.arange(n)
    cats = []; sc = []; m = []; ig = []; np_ = np.zeros(80)
    for j, r in enumerate(recs):
        sel = np.where(assign == j)[0]
        if len(sel) == 0: continue
        cfull = np.bincount(idx, minlength=n) * (assign == j); per = cfull[r['img']]; rep = np.repeat(np.arange(len(r['img'])), per)
        cats.append(r['cat'][rep]); sc.append(r['score'][rep]); m.append(r['match'][:, rep]); ig.append(r['ign'][:, rep]); np_ += (r['npig'][:, sel] * cfull[sel]).sum(1)
    return ap_from(np.concatenate(cats), np.concatenate(sc), np.concatenate(m, 1), np.concatenate(ig, 1), np_)
recs = {}; AP = {}
for k in keys:
    nimg = len(set(d['image_id'] for d in R[k])); print('  (%s: %d images with detections)' % (k, nimg)) if nimg != n else None
    recs[k] = rec_of(R[k]); AP[k] = full_ap(R[k]); print('%-18s %.4f' % (k, AP[k]))
rng = np.random.default_rng(0); boots = [rng.integers(0, n, n) for _ in range(300)]; z = np.zeros(n, int)
def diff(k1, k2):
    d = np.array([ap_assign([recs[k1]], z, b) - ap_assign([recs[k2]], z, b) for b in boots]); return AP[k1] - AP[k2], d.std()
print('\nn=%d. Paired differences (bootstrap sd, 300 draws):' % n)
for a, b in [('gn3__gn3_k21', 'gn3__public'), ('gn3__gn3_k21', 'gn3__pooled_k21'), ('gn3__pooled_k21', 'gn3__public'), ('ct3__ct3_k21', 'ct3__public'), ('ct3__ct3_k21', 'ct3__pooled_k21'), ('ct3__pooled_k21', 'ct3__public'),
             ('clean__gn3_k21', 'clean__public'), ('clean__ct3_k21', 'clean__public'), ('clean__pooled_k21', 'clean__public'),
             ('gn1__gn3_k21', 'gn1__public'), ('gn5__gn3_k21', 'gn5__public'), ('ct1__ct3_k21', 'ct1__public')]:
    print('%-16s minus %-18s %+.4f (sd %.4f)' % ((a, b) + diff(a, b)))
# realised route with the measured router rates (native-resolution noise floor at 0.25% clean FA: recall s1 0.82, s3 1.00, s5 1.00; contrast posterior > 0.9: recall s1 0.60, s3 0.97)
rates = {'clean': ('gn3_k21', 0.0025, 'ct3_k21', 0.0025), 'gn1': ('gn3_k21', 0.82), 'gn3': ('gn3_k21', 1.0), 'gn5': ('gn3_k21', 1.0), 'ct1': ('ct3_k21', 0.60), 'ct3': ('ct3_k21', 0.97)}
rr = np.random.default_rng(3); print('\nRealised route (20 draws) vs public, by condition:')
for c, r in rates.items():
    names = ['public'] + list(r[0::2]); p = [1 - sum(r[1::2])] + list(r[1::2]); rl = [recs[f'{c}__{s}'] for s in names]
    ap_r = np.mean([ap_assign(rl, rr.choice(len(names), n, p=p)) for _ in range(20)])
    print('%-6s public %.4f route %.4f (%+.4f)' % (c, AP[f'{c}__public'], ap_r, ap_r - AP[f'{c}__public']))
