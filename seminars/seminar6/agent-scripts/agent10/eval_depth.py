"""Evaluate the probe: dense AP per (d,K) on the subset; per-image AP proxy; oracle / share-null / signal routes
between a cheap point and the full point. One thread. Usage: eval_depth.py probe.npz"""
import os, sys, json, io, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

z = np.load(sys.argv[1]); n = int(z["n"])
sig = json.loads(str(z["signals"])); T = json.loads(str(z["timings"]))
ids = [r["id"] for r in sig]
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO("/data/datasets/coco/annotations/instances_val2017.json")
keys = sorted([k for k in z.files if k.startswith("d")], key=lambda k: (int(k.split("_K")[1]), int(k[1:].split("_")[0])))
dets = {k: z[k] for k in keys}

def to_list(a):
    return [{"image_id": int(r[0]), "category_id": int(r[1]), "score": float(r[2]), "bbox": [float(v) for v in r[3:7]]} for r in a]

def evaluate(a, img_ids, per_image=False):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(to_list(a))
        E = COCOeval(gt, dt, "bbox"); E.params.imgIds = img_ids
        E.evaluate(); E.accumulate(); E.summarize()
    out = {"AP": E.stats[0], "AP50": E.stats[1], "AP75": E.stats[2], "APs": E.stats[3], "APm": E.stats[4], "APl": E.stats[5]}
    if per_image:
        # per-image AP proxy: COCO AP computed with this image alone (images without GT get nan)
        pi = {}
        for i in img_ids:
            with contextlib.redirect_stdout(io.StringIO()):
                E2 = COCOeval(gt, dt, "bbox"); E2.params.imgIds = [i]; E2.evaluate(); E2.accumulate(); E2.summarize()
            pi[i] = E2.stats[0]
        out["per_image"] = pi
    return out

print(f"n={n} timings(s) {T}")
res = {}
for k in keys:
    res[k] = evaluate(dets[k], ids, per_image=(k in sys.argv[2:]))
    print(f"{k:10s} AP {res[k]['AP']:.4f} AP50 {res[k]['AP50']:.4f} AP75 {res[k]['AP75']:.4f} S/M/L {res[k]['APs']:.3f}/{res[k]['APm']:.3f}/{res[k]['APl']:.3f}", flush=True)
json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_image"} for k, v in res.items()},
          open(sys.argv[1].replace(".npz", "_dense.json"), "w"), indent=1)
pi = {k: v["per_image"] for k, v in res.items() if "per_image" in v}
if pi:
    json.dump(pi, open(sys.argv[1].replace(".npz", "_perimage.json"), "w"))
