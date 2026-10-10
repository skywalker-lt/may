"""Paired image bootstrap of LW-DETR-L minus YOLO26-L on the probe subset (same images), at 512 and 640. ONE thread."""
import os, sys, json, io, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
rng = np.random.default_rng(0)
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO("/data/datasets/coco/annotations/instances_val2017.json")
def prep(dt, ids):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dt), "bbox"); E.params.imgIds = ids; E.evaluate()
    return E
def ap_on(E, sel):
    with contextlib.redirect_stdout(io.StringIO()):
        E.params.imgIds = list(sel); E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0]
I = "/data/tmp/ds-yolo/seminar6/inputs/"
for npz, key, yolo in [("lw200_512.npz", "d3_K300", I+"dumps_r2/dumpml_yolo26l_512_coco.json"), ("lw200.npz", "d3_K300", I+"dumps/dump_yolo26l_coco.json")]:
    z = np.load(npz); ids = [r["id"] for r in json.loads(str(z["signals"]))]; S = set(ids)
    a = z[key]; lw = [{"image_id": int(r[0]), "category_id": int(r[1]), "score": float(r[2]), "bbox": [float(v) for v in r[3:7]]} for r in a]
    yo = [r for r in json.load(open(yolo)) if r["image_id"] in S]
    with contextlib.redirect_stdout(io.StringIO()):
        E1 = COCOeval(gt, gt.loadRes(lw), "bbox"); E2 = COCOeval(gt, gt.loadRes(yo), "bbox")
    base = ap_on(E1, ids) - ap_on(E2, ids)
    d = []
    for b in range(40):
        sel = list(rng.choice(ids, len(ids), replace=True)); sel = sorted(set(sel))  # unique-image draw (pycocotools dedups ids)
        d.append(ap_on(E1, sel) - ap_on(E2, sel))
    print(f"{npz}: LW-DETR-L minus YOLO26-L = {base:+.4f}; 40 paired draws mean {np.mean(d):+.4f} sd {np.std(d):.4f} min {min(d):+.4f}", flush=True)
