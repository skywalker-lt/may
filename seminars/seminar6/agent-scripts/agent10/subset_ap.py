"""AP of COCO-json dumps restricted to the probe subset's image ids (paired comparison with the DETR probes). ONE thread."""
import os, sys, json, io, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ids = [r["id"] for r in json.loads(str(np.load(sys.argv[1])["signals"]))]
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO("/data/datasets/coco/annotations/instances_val2017.json")
S = set(ids)
for p in sys.argv[2:]:
    d = [r for r in json.load(open(p)) if r["image_id"] in S]
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(d), "bbox"); E.params.imgIds = ids; E.evaluate(); E.accumulate(); E.summarize()
    print(f"{os.path.basename(p):40s} subset n={len(ids)} AP {E.stats[0]:.4f}", flush=True)
