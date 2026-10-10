"""Run COCOeval.evaluate() once per dump and cache per-image evalImgs (bbox, val2017) for routing mixtures.
CPU, one thread. Usage: eval_cache.py name dumpfile"""
import sys, json, pickle, time, os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io
D = "/data/tmp/ds-yolo/seminar5/inputs/dumps/"
OUT = "/data/tmp/ds-yolo/seminar5/work/agent8/cache/"
os.makedirs(OUT, exist_ok=True)
name, dump = sys.argv[1], sys.argv[2]
t = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(D + "instances_val2017.json")
    dt = gt.loadRes(D + dump)
    E = COCOeval(gt, dt, "bbox")
    E.evaluate()
    E.accumulate()
E.summarize()
print(name, "AP", round(E.stats[0], 4), "time", round(time.time() - t), "s", flush=True)
# keep only what accumulate needs
slim = []
for e in E.evalImgs:
    if e is None:
        slim.append(None)
    else:
        slim.append({k: e[k] for k in ("image_id", "category_id", "aRng", "maxDet", "dtIds", "gtIds", "dtMatches", "dtScores", "gtIgnore", "dtIgnore")})
pickle.dump({"evalImgs": slim, "imgIds": E.params.imgIds, "catIds": E.params.catIds, "stats": E.stats}, open(OUT + name + ".pkl", "wb"), protocol=4)
