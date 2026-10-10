"""Score a one-to-many + NMS dump (abs path) and, when the o2o twin's cache exists, the same-backbone head mixture:
oracle (per-image AP), share null, GT rules, GT-free count rules (o2o own count, n320 ridge), both directions.
Usage: score_o2m.py name dumppath [o2o_cache_name]"""
import os, sys, json, pickle, copy, contextlib, io, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.stats import spearmanr
W = "/data/tmp/ds-yolo/seminar5/work/agent8/"; D = "/data/tmp/ds-yolo/seminar5/inputs/dumps/"
name, dump = sys.argv[1], sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 else None
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(D + "instances_val2017.json")
if not os.path.exists(W + f"cache/{name}.pkl"):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dump); E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate()
    E.summarize()
    slim = [None if e is None else {k: e[k] for k in ("image_id", "category_id", "aRng", "maxDet", "dtIds", "gtIds", "dtMatches", "dtScores", "gtIgnore", "dtIgnore")} for e in E.evalImgs]
    pickle.dump({"evalImgs": slim, "imgIds": E.params.imgIds, "catIds": E.params.catIds, "stats": E.stats}, open(W + f"cache/{name}.pkl", "wb"), protocol=4)
    print(name, "stats", np.round(E.stats[:6], 4), flush=True)
if base is None: sys.exit()
sys.argv = ["mix.py", f"{base},{name}", base, name]
exec(open(W + "mix.py").read())
