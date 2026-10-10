"""Resolution ceiling of mask prototypes, dense and box-routed (CPU, one thread).
For every non-crowd val2017 instance: rasterise the GT polygon in letterbox-640 space, quantise it at stride s
(area average over s x s cells), reconstruct as the CLI does with real prototypes (bilinear upsample, crop to the box,
threshold 0.5), and measure IoU with the full-resolution mask. IoU is accumulated per 64-px tile so that a routed
composite (stride 4 in the selected tiles, stride 8 elsewhere) is a cheap sum. Proxy mask AP = mean over IoU thresholds
0.50:0.95 of the share of instances reconstructed above threshold (a perfect predictor at that resolution), averaged over
categories as COCO does. Tile selection: static top-T of 100 tiles, ranked by the YOLO26-M dump's boxes (no GT), by GT
boxes (oracle), or at random (null)."""
import json, time, numpy as np, cv2
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools import mask as mu
cv2.setNumThreads(1)
D = '/data/tmp/ds-yolo/seminar6/inputs/dumps/'
gt = COCO(D + 'instances_val2017.json')
dets = json.load(open(D + 'dumpml_yolo26m_coco.json'))
by_img = defaultdict(list)
for d in dets: by_img[d['image_id']].append(d)
S, G, TS = 640, 10, 64
THR = np.arange(0.5, 0.951, 0.05)
STR = [4, 8, 16]
rng = np.random.default_rng(0)

def lb(W, H):
    r = min(S / W, S / H); nw, nh = round(W * r), round(H * r)
    return r, (S - nw) / 2, (S - nh) / 2

def tiles_touched(x1, y1, x2, y2):
    tx1, tx2 = int(max(0, x1 // TS)), int(min(G - 1, (x2 - 1e-6) // TS))
    ty1, ty2 = int(max(0, y1 // TS)), int(min(G - 1, (y2 - 1e-6) // TS))
    return [ty * G + tx for ty in range(ty1, ty2 + 1) for tx in range(tx1, tx2 + 1)]

def rank(boxes, mode):
    """boxes: list of (x1,y1,x2,y2,score) in letterbox px. Returns tile scores."""
    c = np.zeros(G * G)
    for x1, y1, x2, y2, s in boxes:
        a = (x2 - x1) * (y2 - y1)
        if mode == 'sm' and a >= 96 * 96: continue
        if mode == 'edge':  # tiles on the box border, any size
            ts = set(tiles_touched(x1, y1, x2, y2)) - set(tiles_touched(x1 + TS, y1 + TS, x2 - TS, y2 - TS) if (x2 - x1 > 2 * TS and y2 - y1 > 2 * TS) else [])
        else:
            ts = tiles_touched(x1, y1, x2, y2)
        for t in ts: c[t] += s
    return c

Ts = [10, 20, 30, 40, 60]
rules = ['det_sm', 'det_edge', 'gt_sm', 'rand']
res = {('dense', s): [] for s in STR}
for ru in rules:
    for T in Ts: res[(ru, T)] = []
cats, sizes = [], []
t0 = time.time()
ids = gt.getImgIds()
for n, iid in enumerate(ids):
    im = gt.loadImgs(iid)[0]; W, H = im['width'], im['height']
    r, px, py = lb(W, H)
    anns = [a for a in gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False)) if isinstance(a['segmentation'], list)]
    if not anns: continue
    detb = [(d['bbox'][0] * r + px, d['bbox'][1] * r + py, (d['bbox'][0] + d['bbox'][2]) * r + px, (d['bbox'][1] + d['bbox'][3]) * r + py, d['score']) for d in by_img[iid]]
    gtb = [(a['bbox'][0] * r + px, a['bbox'][1] * r + py, (a['bbox'][0] + a['bbox'][2]) * r + px, (a['bbox'][1] + a['bbox'][3]) * r + py, 1.0) for a in anns]
    sel = {}
    for T in Ts:
        sel[('det_sm', T)] = set(np.argsort(-rank(detb, 'sm'), kind='stable')[:T].tolist())
        sel[('det_edge', T)] = set(np.argsort(-rank(detb, 'edge'), kind='stable')[:T].tolist())
        sel[('gt_sm', T)] = set(np.argsort(-rank(gtb, 'sm'), kind='stable')[:T].tolist())
        sel[('rand', T)] = set(rng.permutation(G * G)[:T].tolist())
    for a in anns:
        bx1, by1, bx2, by2 = a['bbox'][0] * r + px, a['bbox'][1] * r + py, (a['bbox'][0] + a['bbox'][2]) * r + px, (a['bbox'][1] + a['bbox'][3]) * r + py
        m = 32  # margin, multiple of 16
        X0 = max(0, int(bx1 // 16) * 16 - m); Y0 = max(0, int(by1 // 16) * 16 - m)
        X1 = min(S, int(np.ceil(bx2 / 16)) * 16 + m); Y1 = min(S, int(np.ceil(by2 / 16)) * 16 + m)
        w, h = X1 - X0, Y1 - Y0
        polys = [[(v * r + (px if i % 2 == 0 else py)) - (X0 if i % 2 == 0 else Y0) for i, v in enumerate(p)] for p in a['segmentation'] if len(p) >= 6]
        if not polys: continue
        ref = mu.decode(mu.merge(mu.frPyObjects(polys, h, w))).astype(bool)
        if ref.sum() == 0: continue
        boxm = np.zeros((h, w), bool)
        boxm[max(0, int(np.floor(by1 - Y0))):int(np.ceil(by2 - Y0)), max(0, int(np.floor(bx1 - X0))):int(np.ceil(bx2 - X0))] = True
        # tile id per pixel of the crop
        tyy = ((np.arange(Y0, Y1)) // TS)[:, None]; txx = ((np.arange(X0, X1)) // TS)[None, :]
        tid = (tyy * G + txx).ravel()
        f = ref.astype(np.float32)
        per = {}
        for s in STR:
            q = f.reshape(h // s, s, w // s, s).mean((1, 3))
            rec = cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR) > 0.5
            rec &= boxm
            I = np.bincount(tid, (rec & ref).ravel(), minlength=G * G); U = np.bincount(tid, (rec | ref).ravel(), minlength=G * G)
            per[s] = (I, U)
            res[('dense', s)].append(I.sum() / U.sum())
        I4, U4 = per[4]; I8, U8 = per[8]
        for key, st in sel.items():
            msk = np.zeros(G * G, bool); msk[list(st)] = True
            res[key].append((np.where(msk, I4, I8).sum()) / np.where(msk, U4, U8).sum())
        cats.append(a['category_id']); sizes.append(a['area'])
    if n % 500 == 0: print(f'{n} images, {len(cats)} instances, {time.time() - t0:.0f}s', flush=True)

cats = np.array(cats); sizes = np.array(sizes)
szb = {'S': sizes < 32 ** 2, 'M': (sizes >= 32 ** 2) & (sizes < 96 ** 2), 'L': sizes >= 96 ** 2}
def ap(iou, idx=None):
    iou = np.array(iou); c = cats if idx is None else cats[idx]; v = iou if idx is None else iou[idx]
    per = [np.mean([(v[c == k] >= t).mean() for t in THR]) for k in np.unique(c)]
    return np.mean(per)
print(f'instances {len(cats)}: S {szb["S"].mean():.3f} M {szb["M"].mean():.3f} L {szb["L"].mean():.3f}')
print(f'{"config":16s} {"proxyAP":>8s} {"S":>7s} {"M":>7s} {"L":>7s} {"meanIoU":>8s}')
for key in res:
    v = np.array(res[key])
    print(f'{str(key):16s} {ap(v):8.4f} {ap(v, szb["S"]):7.4f} {ap(v, szb["M"]):7.4f} {ap(v, szb["L"]):7.4f} {v.mean():8.4f}', flush=True)
np.savez('/data/tmp/ds-yolo/seminar6/work/agent7/proto_ceiling.npz', cats=cats, sizes=sizes, **{f'{k[0]}_{k[1]}': np.array(v) for k, v in res.items()})
