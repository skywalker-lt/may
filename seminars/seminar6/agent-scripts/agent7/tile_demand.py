"""Per-image demand for stride-4 tiles: tiles touched by small/medium (letterbox area < 96^2) YOLO26-M detections
at score >= 0.25 and >= 0.05, and by small/medium GT instances. CPU, one thread."""
import json, numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
D = '/data/tmp/ds-yolo/seminar6/inputs/dumps/'
gt = COCO(D + 'instances_val2017.json'); dets = json.load(open(D + 'dumpml_yolo26m_coco.json'))
by = defaultdict(list)
for d in dets: by[d['image_id']].append(d)
S, G, TS = 640, 10, 64
def tiles(b, r, px, py):
    x1, y1 = b[0] * r + px, b[1] * r + py; x2, y2 = x1 + b[2] * r, y1 + b[3] * r
    if (x2 - x1) * (y2 - y1) >= 96 * 96: return set()
    return {ty * G + tx for ty in range(int(y1 // TS), int(min(G - 1, (y2 - 1e-6) // TS)) + 1) for tx in range(int(x1 // TS), int(min(G - 1, (x2 - 1e-6) // TS)) + 1)}
out = {'det>=0.25': [], 'det>=0.05': [], 'gt': []}
for iid in gt.getImgIds():
    im = gt.loadImgs(iid)[0]; W, H = im['width'], im['height']; r = min(S / W, S / H); px, py = (S - round(W * r)) / 2, (S - round(H * r)) / 2
    for k, thr in [('det>=0.25', 0.25), ('det>=0.05', 0.05)]:
        u = set()
        for d in by[iid]:
            if d['score'] >= thr: u |= tiles(d['bbox'], r, px, py)
        out[k].append(len(u))
    u = set()
    for a in gt.loadAnns(gt.getAnnIds(imgIds=iid, iscrowd=False)): u |= tiles(a['bbox'], r, px, py)
    out['gt'].append(len(u))
for k, v in out.items():
    v = np.array(v)
    print(f'{k:10s} S/M tiles per image: mean {v.mean():.1f} median {np.median(v):.0f} p90 {np.percentile(v, 90):.0f} p99 {np.percentile(v, 99):.0f} max {v.max()}; share > 20: {(v > 20).mean():.3f}, > 30: {(v > 30).mean():.3f}, > 40: {(v > 40).mean():.3f}')
