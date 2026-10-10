# Per-variant pycocotools evaluate() of the YOLO26-M multi-label dump with pyramid-level proxies:
# a variant drops detections that a skipped level would have produced (box size at the 640 model-input scale).
# evalImgs are per (category, area range, image), so any per-image routing between two variants is exact by
# recombining evalImgs and calling accumulate() only.
import json, pickle, sys, time, numpy as np, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'; W = '/data/tmp/ds-yolo/seminar5/work/agent3/'
gt = COCO(D + 'instances_val2017.json')
dets = json.load(open(D + sys.argv[1]))
tag = sys.argv[2]
scale = {i: 640.0 / max(gt.imgs[i]['width'], gt.imgs[i]['height']) for i in gt.imgs}
def sz(d): return np.sqrt(max(d['bbox'][2] * d['bbox'][3], 0)) * scale[d['image_id']]
variants = {'full': lambda s: True,
            'noP3_16': lambda s: s >= 16, 'noP3_32': lambda s: s >= 32, 'noP3_48': lambda s: s >= 48,
            'noP5_96': lambda s: s <= 96, 'noP5_128': lambda s: s <= 128,
            'P4only_32_128': lambda s: 32 <= s <= 128}
want = sys.argv[3].split(',') if len(sys.argv) > 3 else list(variants)
for v in want:
    t0 = time.time(); keep = [d for d in dets if variants[v](sz(d))]
    dt = gt.loadRes(keep) if keep else None
    E = COCOeval(gt, dt, 'bbox'); E.evaluate()
    pickle.dump({'evalImgs': E.evalImgs, 'imgIds': E.params.imgIds, 'catIds': E.params.catIds,
                 'ndet': len(keep)}, open(W + f'ei_{tag}_{v}.pkl', 'wb'))
    E.accumulate(); E.summarize()
    print(f'VARIANT {tag} {v} ndet={len(keep)} AP={E.stats[0]:.4f} S/M/L={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f} ({time.time()-t0:.0f}s)', flush=True)
