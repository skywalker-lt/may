"""Lens-3 check: the N@640 content rectangle (union of YOLO26-N boxes with score >= 0.05, +4% margin, sides rounded up to
32 px, 128 px minimum) against ground truth on full val2017: fraction of GT instances (non-crowd) whose box centre falls
outside the rectangle, by COCO size class; rectangle area as a fraction of the image."""
import json, numpy as np
from common import GT
gt = json.load(open(GT)); ims = {im['id']: im for im in gt['images']}
N = json.load(open('/data/tmp/ds-yolo/seminar6/inputs/dumps/dump_yolo26n_coco.json'))
box = {}
for d in N:
    if d['score'] >= 0.05: box.setdefault(d['image_id'], []).append(d['bbox'])
rect = {}; frac = []
for iid, im in ims.items():
    W, H = im['width'], im['height']
    if iid not in box: rect[iid] = None; continue
    b = np.array(box[iid]); x0, y0 = b[:, 0].min(), b[:, 1].min(); x1, y1 = (b[:, 0] + b[:, 2]).max(), (b[:, 1] + b[:, 3]).max()
    mw, mh = 0.04 * (x1 - x0), 0.04 * (y1 - y0); x0, y0, x1, y1 = max(0, x0 - mw), max(0, y0 - mh), min(W, x1 + mw), min(H, y1 + mh)
    w = max(128, int(np.ceil((x1 - x0) / 32) * 32)); h = max(128, int(np.ceil((y1 - y0) / 32) * 32))
    x1, y1 = min(W, x0 + w), min(H, y0 + h); rect[iid] = (x0, y0, x1, y1); frac.append((x1 - x0) * (y1 - y0) / (W * H))
print('images with a rectangle %d / %d; mean area fraction %.3f, median %.3f, full-frame (>0.98) %.3f' % (len(frac), len(ims), np.mean(frac), np.median(frac), np.mean(np.array(frac) > 0.98)))
out = {k: [0, 0] for k in ['S', 'M', 'L']}; outside_imgs = set()
for a in gt['annotations']:
    if a.get('iscrowd', 0): continue
    k = 'S' if a['area'] < 32 ** 2 else 'M' if a['area'] < 96 ** 2 else 'L'
    r = rect[a['image_id']]; cx, cy = a['bbox'][0] + a['bbox'][2] / 2, a['bbox'][1] + a['bbox'][3] / 2
    out[k][1] += 1
    if r is None or not (r[0] <= cx <= r[2] and r[1] <= cy <= r[3]): out[k][0] += 1; outside_imgs.add(a['image_id'])
for k in out: print('GT %s instances with centre outside the N rectangle: %d / %d = %.3f' % (k, out[k][0], out[k][1], out[k][0] / out[k][1]))
print('total outside %.3f; images with at least one GT outside: %d (%.3f)' % (sum(v[0] for v in out.values()) / sum(v[1] for v in out.values()), len(outside_imgs), len(outside_imgs) / len(ims)))
