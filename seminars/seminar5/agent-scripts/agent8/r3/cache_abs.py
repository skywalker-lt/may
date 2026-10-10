"""Round 3: cache per-image COCOeval evalImgs for a dump given by absolute path (bbox, val2017). CPU, one thread.
Usage: cache_abs.py name /abs/dump.json"""
import os, sys, pickle, time, contextlib, io
os.environ["OMP_NUM_THREADS"] = "1"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
OUT = "/data/tmp/ds-yolo/seminar5/work/agent8/cache/"
name, dump = sys.argv[1], sys.argv[2]; t = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO("/data/tmp/ds-yolo/seminar5/inputs/dumps/instances_val2017.json"); dt = gt.loadRes(dump)
    E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate()
E.summarize()
print(name, "AP", round(E.stats[0], 4), "S/M/L", [round(x, 4) for x in E.stats[3:6]], "time", round(time.time() - t), "s", flush=True)
slim = [None if e is None else {k: e[k] for k in ("image_id", "category_id", "aRng", "maxDet", "dtIds", "gtIds", "dtMatches", "dtScores", "gtIgnore", "dtIgnore")} for e in E.evalImgs]
pickle.dump({"evalImgs": slim, "imgIds": E.params.imgIds, "catIds": E.params.catIds, "stats": E.stats}, open(OUT + name + ".pkl", "wb"), protocol=4)
