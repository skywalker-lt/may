"""Crop-or-shrink ceiling: per image, either shrink the whole image to long side 512 (rect 512x384 for 4:3) or run a
native-resolution 0.8x0.8 crop window of the same engine shape (same cost).  The crop option is scored from the L@640 dump
restricted to the window (detections whose centre lies outside are dropped; boxes clipped), which is what a 512x384 engine
would see at native pixel density up to the padding context.  The window is chosen by an oracle (GT): the 3x3-grid window
holding every GT box centre if one exists, else shrink.  This bounds the route from above."""
import json, numpy as np, io, contextlib, time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
IN = '/data/tmp/ds-yolo/seminar6/inputs'
gt = COCO(f'{IN}/dumps/instances_val2017.json'); imgIds = sorted(gt.getImgIds())
info = {im['id']: (im['width'], im['height']) for im in gt.loadImgs(imgIds)}
anns = {i: [] for i in imgIds}
for a in gt.loadAnns(gt.getAnnIds(iscrowd=False)): anns[a['image_id']].append(a['bbox'])
F = 0.8
def windows(w, h):
    cw, ch = w * F, h * F
    return [(x, y, cw, ch) for x in np.linspace(0, w - cw, 3) for y in np.linspace(0, h - ch, 3)]
def inside(b, win):
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
    return win[0] <= cx <= win[0] + win[2] and win[1] <= cy <= win[1] + win[3]
choice = {}
for i in imgIds:
    w, h = info[i]; bs = anns[i]
    if not bs: choice[i] = None; continue
    for win in windows(w, h):
        if all(inside(b, win) for b in bs): choice[i] = win; break
    else: choice[i] = None
ncrop = sum(v is not None for v in choice.values())
print(f'images whose GT centres all fit one 0.8x0.8 window: {ncrop} of {len(imgIds)} (empty-GT images count as shrink)')
d640 = json.load(open(f'{IN}/dumps/dump_yolo26l_coco.json')); d512 = json.load(open(f'{IN}/dumps_r2/dumpml_yolo26l_512_coco.json'))
def evaluate(dets, label):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), 'bbox'); E.params.imgIds = imgIds; E.evaluate(); E.accumulate(); E.summarize()
    print(f'{label}: AP {E.stats[0]:.4f} S/M/L {E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f}', flush=True)
mixed = [d for d in d512 if choice[d['image_id']] is None]
for d in d640:
    win = choice[d['image_id']]
    if win is None: continue
    b = d['bbox']
    if not inside(b, win): continue
    x0, y0 = max(b[0], win[0]), max(b[1], win[1]); x1, y1 = min(b[0] + b[2], win[0] + win[2]), min(b[1] + b[3], win[1] + win[3])
    mixed.append({**d, 'bbox': [x0, y0, x1 - x0, y1 - y0]})
evaluate(mixed, f'oracle crop-or-shrink (crop on {ncrop} images, L@640 dets in window; shrink L@512 elsewhere)')
# the same images, but shrink everywhere (what the crop replaces): L@512 is 0.5262 by the dump; and the crop images alone under L@640
sub = [d for d in d640 if choice[d['image_id']] is not None] + [d for d in d512 if choice[d['image_id']] is None]
evaluate(sub, 'same images at full L@640 instead of the crop (upper bound of what the crop can give)')
