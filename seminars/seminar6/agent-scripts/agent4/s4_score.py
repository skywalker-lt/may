"""Score the cross matrix: AP per (condition, BN set) on the eval subset (official pycocotools on the subset), paired
bootstrap over images (fast AP from cached match records) for the differences that matter, and the realised route
(router_pred_s3.json) against the oracle route, the pooled set and the public set."""
import sys, json, io, contextlib, numpy as np
sys.path.insert(0, '/data/tmp/ds-yolo/seminar6/work/agent4'); from fastap import ap_from
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from common import GT, OUT
with contextlib.redirect_stdout(io.StringIO()): gt = COCO(GT)
E = json.load(open(f'{OUT}/eval_ids.json')); sub = E['sub']; n = len(sub); ii = {im: i for i, im in enumerate(sub)}
catIds = sorted(gt.getCatIds()); ci = {c: k for k, c in enumerate(catIds)}
R = json.load(open(f'{OUT}/matrix_dets.json'))
def rec_of(dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dets) if dets else None
        Ev = COCOeval(gt, dt, 'bbox'); Ev.params.imgIds = sub; Ev.params.areaRng = [[0, 1e10]]; Ev.params.areaRngLbl = ['all']; Ev.params.maxDets = [100]; Ev.evaluate()
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
    return Ev.stats[0], Ev.stats[1], Ev.stats[2]
def ap_assign(recs, assign, idx=None):
    """assign: (n,) record index per image; idx: multiset of image indices (bootstrap) or None."""
    if idx is None: idx = np.arange(n)
    cats = []; sc = []; m = []; ig = []; np_ = np.zeros(80)
    for j, r in enumerate(recs):
        sel_imgs = np.where(assign == j)[0]
        if len(sel_imgs) == 0: continue
        cfull = np.bincount(idx, minlength=n) * (assign == j); cnt = cfull[sel_imgs]
        per = cfull[r['img']]
        rep = np.repeat(np.arange(len(r['img'])), per)
        cats.append(r['cat'][rep]); sc.append(r['score'][rep]); m.append(r['match'][:, rep]); ig.append(r['ign'][:, rep])
        np_ += (r['npig'][:, sel_imgs] * cnt).sum(1)
    return ap_from(np.concatenate(cats), np.concatenate(sc), np.concatenate(m, 1), np.concatenate(ig, 1), np_)
conds = ['clean', 'gn3', 'db3', 'ct3', 'br3']; fam_sets = ['gn3', 'db3', 'ct3', 'br3']
recs = {k: rec_of(v) for k, v in R.items() if v}
n_done = len(set(d['image_id'] for d in R['clean__public'])); assert n_done == n, (n_done, n)
AP = {}
print('eval subset n=%d val2017 images (seed 0). Official pycocotools AP / AP50 / AP75:' % n)
for k in R:
    if R[k]: a = full_ap(R[k]); AP[k] = a[0]; print('  %-20s %.4f / %.4f / %.4f' % (k, *a))
pub = [d for d in json.load(open('/data/tmp/ds-yolo/seminar6/inputs/dumps/dumpml_yolo26m_coco.json')) if d['image_id'] in ii]
print('parity: public YOLO26-M protocol dump on the same subset %.4f vs CPU fp32 clean__public %.4f' % (full_ap(pub)[0], AP['clean__public']))
rng = np.random.default_rng(0); B = 200; boots = [rng.integers(0, n, n) for _ in range(B)]
z = np.zeros(n, int)
def diff(k1, k2):
    d = np.array([ap_assign([recs[k1]], z, b) - ap_assign([recs[k2]], z, b) for b in boots]); return AP[k1] - AP[k2], d.std()
print('\nPaired differences (bootstrap sd over images, %d draws):' % B)
for c in fam_sets:
    print('%s: own minus public %+.4f (sd %.4f) | pooled minus public %+.4f (sd %.4f) | own minus pooled %+.4f (sd %.4f)' % ((c,) + diff(f'{c}__{c}', f'{c}__public') + diff(f'{c}__pooled', f'{c}__public') + diff(f'{c}__{c}', f'{c}__pooled')))
print('clean: misroute tax (set minus public): ' + ' '.join('%s %+.4f (sd %.4f)' % ((s,) + diff(f'clean__{s}', 'clean__public')) for s in ['cleanrecal', 'pooled'] + fam_sets))
def mpc_of(pick):
    return np.mean([AP[f'{c}__{pick(c)}'] for c in fam_sets])
print('\nmPC over the four severity-3 conditions: public %.4f | pooled %.4f | own (oracle family) %.4f' % (mpc_of(lambda c: 'public'), mpc_of(lambda c: 'pooled'), mpc_of(lambda c: c)))
# realised route: router decisions on these images (out-of-fold), clean-prior tau in {0.0, 0.7}
for tau in ('0.0', '0.7'):
    P = json.load(open(f'{OUT}/router_pred_s3_tau{tau}.json')); fams = P['fams']; dec = {}
    for im, t, p, s in zip(P['img'], P['true'], P['pred'], P['sev']): dec[(im, fams[t] + ('3' if s else ''))] = fams[p] + ('3' if p else '')
    out = []
    for c in conds:
        avail = [k.split('__')[1] for k in R if k.startswith(c + '__')]
        a = []
        for im in sub:
            d = dec[(im, c)]; d = 'public' if d == 'clean3' or d == 'clean' else d
            a.append(avail.index(d) if d in avail else avail.index('public'))  # misroute to a set not evaluated under c: scored as public
        a = np.array(a); rl = [recs[f'{c}__{s}'] for s in avail]
        out.append((c, ap_assign(rl, a), (np.array(avail)[a] == ('public' if c == 'clean' else c)).mean()))
    print('router tau %s: ' % tau + ' | '.join('%s %.4f (acc %.2f)' % o for o in out) + ' | mPC %.4f' % np.mean([o[1] for o in out[1:]]))
# Calibration: ECE of detection confidence against precision at IoU 0.5 (dets with score >= 0.05, 10 equal-width bins)
def ece(r, thr=0.05):
    s = r['score']; ok = (s >= thr) & ~r['ign'][0]; s = s[ok]; m = r['match'][0][ok].astype(float)
    bins = np.minimum((s * 10).astype(int), 9); e = 0.0
    for b in range(10):
        q = bins == b
        if q.sum() == 0: continue
        e += q.mean() * abs(s[q].mean() - m[q].mean())
    return e, s.mean(), m.mean()
print('\nCalibration (ECE at IoU 0.5 over dets with score >= 0.05 | mean score | precision):')
for c in conds:
    print('%-6s ' % c + ' | '.join('%s: ECE %.3f (%.2f/%.2f)' % ((s,) + ece(recs[f'{c}__{s}'])) for s in (['public', 'pooled'] if c == 'clean' else ['public', 'pooled', c])))
