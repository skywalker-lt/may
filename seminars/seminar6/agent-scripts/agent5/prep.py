"""Evaluate each dump once with pycocotools (multi-label protocol) and cache evalImgs for per-image mixing.
CPU, one thread (OMP_NUM_THREADS=1 set by the caller)."""
import json, pickle, sys, time, os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

IN = '/data/tmp/ds-yolo/seminar6/inputs'
OUT = '/data/tmp/ds-yolo/seminar6/work/agent5/cache'
os.makedirs(OUT, exist_ok=True)
DUMPS = {
    'l448': f'{IN}/dumps_r2/dumpml_yolo26l_448_coco.json',
    'l512': f'{IN}/dumps_r2/dumpml_yolo26l_512_coco.json',
    'l544': f'{IN}/dumps_r2/dumpml_yolo26l_544_coco.json',
    'l576': f'{IN}/dumps_r2/dumpml_yolo26l_576_coco.json',
    'l640': f'{IN}/dumps/dump_yolo26l_coco.json',
    'm448': f'{IN}/dumps_r2/dumpml_yolo26m_448_coco.json',
    'm512': f'{IN}/dumps/dumpml_yolo26m_512_coco.json',
    'm576': f'{IN}/dumps_r2/dumpml_yolo26m_576_coco.json',
    'm608': f'{IN}/dumps_r2/dumpml_yolo26m_608_coco.json',
    'm640': f'{IN}/dumps/dumpml_yolo26m_coco.json',
}
gt = COCO(f'{IN}/dumps/instances_val2017.json')
names = sys.argv[1:] or list(DUMPS)
for name in names:
    t0 = time.time()
    dt = gt.loadRes(DUMPS[name])
    E = COCOeval(gt, dt, 'bbox')
    E.params.imgIds = sorted(gt.getImgIds())
    E.evaluate()
    E.accumulate()
    E.summarize()
    with open(f'{OUT}/{name}.pkl', 'wb') as f:
        pickle.dump({'evalImgs': E.evalImgs, 'params': E.params, 'stats': E.stats}, f, protocol=4)
    print(f'{name} AP={E.stats[0]:.4f} S/M/L={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f} ({time.time()-t0:.0f}s)', flush=True)
