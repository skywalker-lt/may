"""Agent 10 round 1: ceiling of a per-query refiner bound to a base's own queries (a DETR-style decoder on YOLO26's
one-to-one top-300). 'Teacher-quality refiner': each base detection is matched one-to-one (greedy by base score, same
class, IoU >= thr) to a teacher detection and takes the teacher's box and score; unmatched base detections keep their
box and get score * u. Recall stays bounded by the base's queries. ONE thread. Usage: base.json teacher.json thr u"""
import os, sys, json, io, contextlib
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from collections import defaultdict
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
base, teach, thr, u = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
def load(p):
    d = defaultdict(list)
    for r in json.load(open(p)): d[r["image_id"]].append(r)
    return d
B, T = load(base), load(teach)
def iou(b, bs):
    x1 = np.maximum(b[0], bs[:,0]); y1 = np.maximum(b[1], bs[:,1]); x2 = np.minimum(b[0]+b[2], bs[:,0]+bs[:,2]); y2 = np.minimum(b[1]+b[3], bs[:,1]+bs[:,3])
    inter = (x2-x1).clip(0)*(y2-y1).clip(0); return inter/(b[2]*b[3]+bs[:,2]*bs[:,3]-inter+1e-9)
out, matched, tot = [], 0, 0
for i, rows in B.items():
    rows = sorted(rows, key=lambda r: -r["score"]); tr = T.get(i, [])
    tb = np.array([r["bbox"] for r in tr]) if tr else np.zeros((0,4)); tc = np.array([r["category_id"] for r in tr]); used = np.zeros(len(tr), bool)
    for r in rows:
        tot += 1
        if len(tr):
            ok = (tc == r["category_id"]) & ~used
            if ok.any():
                v = np.where(ok, iou(r["bbox"], tb), -1); j = int(v.argmax())
                if v[j] >= thr:
                    used[j] = True; matched += 1
                    out.append({"image_id": i, "category_id": r["category_id"], "bbox": tr[j]["bbox"], "score": tr[j]["score"]}); continue
        out.append({**r, "score": r["score"] * u})
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO("/data/datasets/coco/annotations/instances_val2017.json")
    E = COCOeval(gt, gt.loadRes(out), "bbox"); E.evaluate(); E.accumulate(); E.summarize()
print(f"{os.path.basename(base)} refined to {os.path.basename(teach)} (IoU>={thr}, unmatched score x{u}): matched {matched/tot:.3f} of {tot} | AP {E.stats[0]:.4f} AP50 {E.stats[1]:.4f} AP75 {E.stats[2]:.4f}", flush=True)
