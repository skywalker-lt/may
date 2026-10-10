# Round 2: per-variant evaluate() of a dump, level proxies computed at the dump's own model-input scale.
# usage: build2.py <dump json (abs or in dumps/)> <tag> <input scale px> <variants comma list>
import json, pickle, sys, time, numpy as np, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = '/data/tmp/ds-yolo/seminar5/inputs/dumps/'; W = '/data/tmp/ds-yolo/seminar5/work/agent3/'
gt = COCO(D + 'instances_val2017.json')
p = sys.argv[1] if sys.argv[1].startswith('/') else D + sys.argv[1]
dets = json.load(open(p)); tag = sys.argv[2]; S = float(sys.argv[3])
scale = {i: S / max(gt.imgs[i]['width'], gt.imgs[i]['height']) for i in gt.imgs}
def sz(d): return np.sqrt(max(d['bbox'][2] * d['bbox'][3], 0)) * scale[d['image_id']]
variants = {'full': lambda s: True, 'noP3_16': lambda s: s >= 16, 'noP3_32': lambda s: s >= 32}
for v in sys.argv[4].split(','):
    t0 = time.time(); keep = [d for d in dets if variants[v](sz(d))]
    E = COCOeval(gt, gt.loadRes(keep), 'bbox'); E.evaluate()
    pickle.dump({'evalImgs': E.evalImgs, 'imgIds': E.params.imgIds, 'catIds': E.params.catIds, 'ndet': len(keep)},
                open(W + f'ei_{tag}_{v}.pkl', 'wb'))
    E.accumulate(); E.summarize()
    print(f'VARIANT {tag} {v} ndet={len(keep)} AP={E.stats[0]:.4f} S/M/L={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f} ({time.time()-t0:.0f}s)', flush=True)
