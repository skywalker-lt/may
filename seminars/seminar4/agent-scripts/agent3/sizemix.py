#!/usr/bin/env python3
"""Agent 3, round 2: box-level (not image-level) resolution mixture, no ground truth used. Detections of yolo26m at
768 are kept when their box area is below a threshold A, detections at 640 (or 512) otherwise; per-box union is a
realisable rule for a pixel-level resolution design, and the upper bound a per-pixel router could reach.
Scored with pycocotools on full val2017, multi-label dumps."""
import json, os, sys, numpy as np, contextlib, io
os.environ["OMP_NUM_THREADS"] = "2"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
gt = COCO(f"{D}/instances_val2017.json")
def ap(dets):
    ev = COCOeval(gt, gt.loadRes(dets), "bbox")
    with contextlib.redirect_stdout(io.StringIO()): ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev.stats[0], ev.stats[3], ev.stats[4], ev.stats[5]
m640 = json.load(open(f"{D}/dumpml_yolo26m_coco.json")); m768 = json.load(open(f"{D}/dumpml_yolo26m_768_coco.json"))
m512 = json.load(open(f"{D}/dumpml_yolo26m_512_coco.json"))
area = lambda d: d["bbox"][2] * d["bbox"][3]
for lo_name, lo in (("640", m640), ("512", m512)):
    for thr in (32, 48, 64, 96):
        mix = [d for d in m768 if area(d) < thr ** 2] + [d for d in lo if area(d) >= thr ** 2]
        a = ap(mix); print(f"small<{thr}px from 768, rest from {lo_name}: AP {a[0]:.4f} S {a[1]:.3f} M {a[2]:.3f} L {a[3]:.3f}", flush=True)
# the other direction as a null: small from 640, rest from 768
mix = [d for d in m640 if area(d) < 48 ** 2] + [d for d in m768 if area(d) >= 48 ** 2]
a = ap(mix); print(f"NULL small<48px from 640, rest from 768: AP {a[0]:.4f} S {a[1]:.3f} M {a[2]:.3f} L {a[3]:.3f}")
