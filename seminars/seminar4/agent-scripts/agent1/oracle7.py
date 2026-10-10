#!/usr/bin/env python3
"""Round 2 (agent 1): resolution routing by a physical rule (object scale), not a ground-truth oracle.
Rule: images with no small GT object (area < 32^2) go to 512; the top-p images by small-object count go to 768; the
rest stay at 640; p set so the mean T4 latency (3.781 / 5.355 / 6.959 ms, insert section 2) is 5.355 ms.
Realistic twin: the same rule on YOLO26-M@512's own boxes (score >= 0.25, area < 32^2), i.e. a cheap-pass signal."""
import json, os, contextlib, io
import numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
gt = COCO(f"{D}/instances_val2017.json"); IMG = sorted(gt.getImgIds()); pos = {i: k for k, i in enumerate(IMG)}
L = np.array([3.781, 5.355, 6.959]); rng = np.random.default_rng(0)
dets = {}
for k, f in (("512", "dumpml_yolo26m_512_coco.json"), ("640", "dump_yolo26m_coco.json"), ("768", "dumpml_yolo26m_768_coco.json")):
    dets[k] = {}
    for d in json.load(open(f"{D}/{f}")): dets[k].setdefault(d["image_id"], []).append(d)
KEYS = ["512", "640", "768"]
def evaluate(choice, tag):
    res = []
    for i, c in zip(IMG, choice): res += dets[KEYS[c]].get(i, [])
    with contextlib.redirect_stdout(io.StringIO()):
        ev = COCOeval(gt, gt.loadRes(res), "bbox"); ev.params.imgIds = IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
    st = ev.stats; sh = np.bincount(choice, minlength=3) / len(IMG)
    print(f"EVAL7 {tag:50s} L={L[choice].mean():.2f} shares={np.round(sh,2).tolist()} AP={st[0]:.4f} S/M/L={st[3]:.4f}/{st[4]:.4f}/{st[5]:.4f}", flush=True)
def small_gt(i):
    return sum(1 for a in gt.loadAnns(gt.getAnnIds(imgIds=i, iscrowd=False)) if a["area"] < 32 ** 2)
def small_det(i):
    return sum(1 for d in dets["512"].get(i, []) if d["score"] >= 0.25 and d["bbox"][2] * d["bbox"][3] < 32 ** 2)
for name, fn in (("GT small-object count rule", small_gt), ("M@512 own small-box count rule", small_det)):
    z = np.array([fn(i) for i in IMG], float)
    c = np.ones(len(IMG), int); c[z == 0] = 0
    order = np.argsort(-z); n512 = (z == 0).sum()
    # largest p such that the mean fits the budget
    for n768 in range(0, len(IMG) - n512 + 1, 25):
        cc = c.copy(); cc[order[:n768]] = 2
        if L[cc].mean() > 5.355: break
        c_ok = cc
    evaluate(c_ok, name); evaluate(rng.permutation(c_ok), "  null random route, same shares")
print("DONE7")
