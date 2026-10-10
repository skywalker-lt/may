"""Score the cached-stage series on its 400-image subset; GT for the shifted frames is shifted by (d, d) in letterboxed px
converted to native px (shift applied to the native image, so native GT moves by d native px, clipped at the border)."""
import json, glob, copy, numpy as np, contextlib, io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
res = {}
for f in sorted(glob.glob("cache_dets_*.json")):
    for k, v in json.load(open(f)).items(): res.setdefault(k, []).extend(v)
with contextlib.redirect_stdout(io.StringIO()): G = COCO("/data/tmp/ds-yolo/seminar6/inputs/dumps/instances_val2017.json")
ids = sorted({d["image_id"] for d in res["full0"]} | set())
import random
def gt_shift(d):
    g = COCO(); ds = copy.deepcopy(G.dataset); W = {i["id"]: (i["width"], i["height"]) for i in ds["images"]}
    anns = []
    for a in ds["annotations"]:
        w, h = W[a["image_id"]]; x, y, bw, bh = a["bbox"]; x1, y1 = min(x + d, w), min(y + d, h); x2, y2 = min(x + bw + d, w), min(y + bh + d, h)
        if x2 - x1 < 1 or y2 - y1 < 1: continue
        a = dict(a, bbox=[x1, y1, x2 - x1, y2 - y1], area=a["area"] * (x2 - x1) * (y2 - y1) / max(bw * bh, 1e-6)); a.pop("segmentation", None); anns.append(a)
    ds["annotations"] = anns; g.dataset = ds
    with contextlib.redirect_stdout(io.StringIO()): g.createIndex()
    return g
sel = sorted({d["image_id"] for d in res["full0"]})
cats = sorted(G.getCatIds())
for k in res:
    for d in res[k]: d["category_id"] = cats[d["category_id"]]
print("images", len(sel))
for cond in sorted(res):
    d = 0 if cond == "full0" else int(cond.split("_")[-1].replace("full", "")) if "_" in cond else int(cond[4:])
    g = gt_shift(d)
    with contextlib.redirect_stdout(io.StringIO()):
        dt = g.loadRes(res[cond]) if res[cond] else None; E = COCOeval(g, dt, "bbox"); E.params.imgIds = sel; E.evaluate(); E.accumulate(); E.summarize()
    print(f"{cond:8s} d={d:2d} AP={E.stats[0]:.4f} AP50={E.stats[1]:.4f} S/M/L={E.stats[3]:.3f}/{E.stats[4]:.3f}/{E.stats[5]:.3f}", flush=True)
