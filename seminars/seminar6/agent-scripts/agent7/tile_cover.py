"""Bound for the tiled mask-prototype route: box AP of YOLO26-M (multi-label dump) restricted to detections lying inside
the top-T of 100 letterbox tiles (64 px at 640) ranked by the detector's own score-weighted box coverage.
Pessimistic: a detection survives only if every tile it touches is selected (mask fully computed).
Optimistic: a detection survives if the tile holding its box centre is selected (mask partially computed).
Also: GT-coverage oracle ranking, random-tile null, person statistics for the pose route, padding tile share.
CPU, one thread."""
import json, sys, time, numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io

D = '/data/tmp/ds-yolo/seminar6/inputs/dumps/'
gt = COCO(D + 'instances_val2017.json')
dets = json.load(open(D + 'dumpml_yolo26m_coco.json'))
print('detections', len(dets))
imgs = {i['id']: i for i in gt.loadImgs(gt.getImgIds())}
G = 10; S = 640

def tiles_of(box, W, H):
    """Tiles (in the 10x10 letterbox grid) touched by an xywh box in original pixels, with intersection fractions."""
    r = S / max(W, H); padx = (S - W * r) / 2; pady = (S - H * r) / 2
    x1 = box[0] * r + padx; y1 = box[1] * r + pady; x2 = x1 + box[2] * r; y2 = y1 + box[3] * r
    ts = S / G
    tx1, tx2 = int(max(0, x1 // ts)), int(min(G - 1, (x2 - 1e-6) // ts)); ty1, ty2 = int(max(0, y1 // ts)), int(min(G - 1, (y2 - 1e-6) // ts))
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    out = []
    for ty in range(ty1, ty2 + 1):
        for tx in range(tx1, tx2 + 1):
            ix = max(0, min(x2, (tx + 1) * ts) - max(x1, tx * ts)); iy = max(0, min(y2, (ty + 1) * ts) - max(y1, ty * ts))
            out.append((ty * G + tx, ix * iy / area))
    cx = int(min(G - 1, ((x1 + x2) / 2) // ts)); cy = int(min(G - 1, ((y1 + y2) / 2) // ts))
    return out, cy * G + cx

def content_tiles(W, H):
    r = S / max(W, H); padx = (S - W * r) / 2; pady = (S - H * r) / 2
    ts = S / G; n = 0
    for ty in range(G):
        for tx in range(G):
            if (tx + 1) * ts > padx and tx * ts < S - padx and (ty + 1) * ts > pady and ty * ts < S - pady: n += 1
    return n

by_img = defaultdict(list)
for d in dets: by_img[d['image_id']].append(d)

# padding share
cont = [content_tiles(imgs[i]['width'], imgs[i]['height']) for i in imgs]
print(f'content tiles per image: mean {np.mean(cont):.1f} of 100 (padding share {1 - np.mean(cont) / 100:.3f})')

# person statistics (GT) and person detections
pid = gt.getCatIds(catNms=['person'])[0]
imgs_with_person = set(a['image_id'] for a in gt.loadAnns(gt.getAnnIds(catIds=[pid], iscrowd=False)))
npers = defaultdict(int)
for a in gt.loadAnns(gt.getAnnIds(catIds=[pid], iscrowd=False)): npers[a['image_id']] += 1
print(f'images with >=1 GT person: {len(imgs_with_person)} of {len(imgs)} ({len(imgs_with_person) / len(imgs):.3f}); persons/image among them mean {np.mean(list(npers.values())):.2f}, p95 {np.percentile(list(npers.values()), 95):.0f}, max {max(npers.values())}')
for thr in [0.001, 0.05, 0.25]:
    c = [sum(1 for d in by_img[i] if d['category_id'] == pid and d['score'] >= thr) for i in imgs]
    print(f'person detections/image at score>={thr}: mean {np.mean(c):.1f}, p95 {np.percentile(c, 95):.0f}, max {max(c)}; images with >=1: {np.mean([x > 0 for x in c]):.3f}')
# rank of the last GT-matched person? keep simple: share of GT persons covered by top-Kp person detections (IoU>=0.5)
def iou(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]; bx2, by2 = b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0])); ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih; return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)
for Kp in [10, 20, 50]:
    hit = tot = 0
    for i in imgs_with_person:
        pd = sorted([d for d in by_img[i] if d['category_id'] == pid], key=lambda d: -d['score'])[:Kp]
        for a in gt.loadAnns(gt.getAnnIds(imgIds=[i], catIds=[pid], iscrowd=False)):
            tot += 1; hit += any(iou(a['bbox'], d['bbox']) >= 0.5 for d in pd)
    print(f'GT persons recalled (IoU>=0.5) by the top-{Kp} person-class anchors: {hit / tot:.4f}')

def evalap(res, tag):
    t = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(res) if res else None
        E = COCOeval(gt, dt, 'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    s = E.stats
    print(f'{tag:40s} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S/M/L {s[3]:.4f}/{s[4]:.4f}/{s[5]:.4f}  n={len(res)}  ({time.time() - t:.0f}s)', flush=True)
    return s[0]

rng = np.random.default_rng(0)
def restrict(T, mode='pess', rank='det', seed=None):
    out = []; kept_score = 0; all_score = 0
    for i, ds in by_img.items():
        W, H = imgs[i]['width'], imgs[i]['height']
        cov = np.zeros(G * G)
        if rank == 'det':
            for d in ds:
                for t, f in tiles_of(d['bbox'], W, H)[0]: cov[t] += d['score'] * f
        elif rank == 'gt':
            for a in gt.loadAnns(gt.getAnnIds(imgIds=[i], iscrowd=False)):
                for t, f in tiles_of(a['bbox'], W, H)[0]: cov[t] += f
        else:
            cov = rng.random(G * G)
        sel = set(np.argsort(-cov)[:T].tolist())
        for d in ds:
            ts, ct = tiles_of(d['bbox'], W, H)
            all_score += d['score']
            ok = all(t in sel for t, _ in ts) if mode == 'pess' else (ct in sel)
            if ok: out.append(d); kept_score += d['score']
    print(f'  [T={T} {mode} {rank}] detections kept {len(out) / len(dets):.3f}, score mass kept {kept_score / all_score:.3f}')
    return out

evalap(dets, 'full dump (dense Proto everywhere)')
for T in [20, 30, 40, 60]:
    evalap(restrict(T, 'pess'), f'top-{T} det-ranked tiles, pessimistic')
for T in [30, 40]:
    evalap(restrict(T, 'opt'), f'top-{T} det-ranked tiles, optimistic')
evalap(restrict(30, 'pess', 'gt'), 'top-30 GT-ranked tiles (oracle), pess')
evalap(restrict(30, 'pess', 'rand'), 'random 30 tiles (null), pess')
