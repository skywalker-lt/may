# Score the CPU level variants of public YOLO26-M on the val2017 subset (every 5th image id; B0 on every 4th of those).
# Variants: A (shipped o2o), H (P3 anchors removed, o2o), Am / Hm (o2m + NMS, all / P4-P5 anchors), B0 (epoch-0 B),
# and the round-1 size proxies tau16 / tau32 built from A on the same images. Mixtures by the m640 small-count router.
import json, gzip, copy, contextlib, io, sys, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'; W = '/data/tmp/ds-yolo/seminar5/work/agent3/'
with contextlib.redirect_stdout(io.StringIO()): gt = COCO(D + 'instances_val2017.json')
recs = json.load(gzip.open(W + 'r2/' + sys.argv[1], 'rt')); ids = sorted(int(k) for k in recs)
scale = {i: 640.0 / max(gt.imgs[i]['width'], gt.imgs[i]['height']) for i in gt.imgs}
def dets(v, keep=lambda d: True, only=None):
    out = []
    for k, r in recs.items():
        if only is not None and int(k) not in only: continue
        for x in r[v]:
            d = {'image_id': int(k), 'bbox': x[:4], 'score': x[4], 'category_id': x[5]}
            if keep(d): out.append(d)
    return out
sz = lambda d: np.sqrt(max(d['bbox'][2] * d['bbox'][3], 0)) * scale[d['image_id']]
def ev(dl, img_ids):
    E = COCOeval(gt, gt.loadRes(dl) if dl else COCO(), 'bbox'); E.params.imgIds = img_ids
    with contextlib.redirect_stdout(io.StringIO()): E.evaluate()
    return E
def acc(E0, evs, mask):
    E = copy.copy(E0); I = len(E0.params.imgIds); A = len(E0.params.areaRng); K = len(E0.params.catIds)
    m = np.tile(mask, K * A); E.evalImgs = [b if not r else a for b, a, r in zip(evs[0].evalImgs, evs[1].evalImgs, m)]
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[[0, 3, 4, 5]]
def alone(E):
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[[0, 3, 4, 5]]
fmt = lambda s: f'{s[0]:.4f} S/M/L {s[1]:.4f}/{s[2]:.4f}/{s[3]:.4f}'
dump = json.load(open(D + 'dumpml_yolo26m_coco.json')); idset = set(ids)
V = {'dumpM': [d for d in dump if d['image_id'] in idset], 'A': dets('A'), 'H': dets('H'), 'Am': dets('Am'), 'Hm': dets('Hm'),
     'tau16': dets('A', lambda d: sz(d) >= 16), 'tau32': dets('A', lambda d: sz(d) >= 32)}
E = {k: ev(v, ids) for k, v in V.items()}
print(f'subset: {len(ids)} images')
for k in V: print(f'alone {k:6s} {fmt(alone(E[k]))}', flush=True)
R = np.load(W + 'router_scores.npz'); rid = {int(i): j for j, i in enumerate(R['ids'])}
key = np.array([R['m640_log1p_small'][rid[i]] for i in ids]); g = np.load(W + 'gt_counts.npz')
gidx = {int(i): j for j, i in enumerate(g['ids'])}; small = np.array([g['C'][gidx[i], 0] + g['C'][gidx[i], 1] for i in ids])
n = len(ids); tb = np.random.default_rng(1).random(n) * 1e-6
for s in (0.4, 0.5):
    for nm, k in [('m640', key), ('GT', small), ('null', np.random.default_rng(7).random(n))]:
        mask = np.zeros(n, bool); mask[np.argsort(k + tb)[:int(round(s * n))]] = True
        line = [f'{v}: {acc(E["A"] if v != "Hm" else E["Am"], [E["A"] if v != "Hm" else E["Am"], E[v]], mask)[0]:.4f}' for v in ('tau16', 'tau32', 'H', 'Hm')]
        print(f'mix share {s:.1f} router {nm:5s} | ' + ' | '.join(line), flush=True)
b0ids = [i for j, i in enumerate(ids) if j % 4 == 0]
print(f'B0 subset: {len(b0ids)} images | A {fmt(alone(ev(dets("A", only=set(b0ids)), b0ids)))} | H {fmt(alone(ev(dets("H", only=set(b0ids)), b0ids)))} | B0 {fmt(alone(ev(dets("B0", only=set(b0ids)), b0ids)))}', flush=True)
