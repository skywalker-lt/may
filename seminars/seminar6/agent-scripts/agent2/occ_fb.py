"""Feedback occupancy rectangle: the rectangle spanned by frame t-1's own L detections (score>=0.05, margin m of the image
side), applied to L@544's detections on frame t (still-image proxy: same image). Nothing outside. AP and mean area fraction."""
import json, numpy as np, contextlib, io, sys
from evlib import *
G = gt(); W = {i["id"]: (i["width"], i["height"]) for i in G.dataset["images"]}
rng = np.random.default_rng(0)
seer = json.load(open(PATHS[sys.argv[1]])); m = float(sys.argv[2]); drop = float(sys.argv[3]); thr = 0.05
base = json.load(open(PATHS["l544"]))
R = {}
for d in seer:
    if d["score"] < thr or rng.random() < drop: continue
    x, y, w, h = d["bbox"]; r = R.setdefault(d["image_id"], [1e9, 1e9, -1e9, -1e9])
    r[0] = min(r[0], x); r[1] = min(r[1], y); r[2] = max(r[2], x + w); r[3] = max(r[3], y + h)
frac = []; out = []
for iid, (w, h) in W.items():
    if iid not in R: frac.append(1.0); continue
    x0, y0, x1, y1 = R[iid]; mx, my = m * w, m * h
    x0 = max(0, x0 - mx); y0 = max(0, y0 - my); x1 = min(w, x1 + mx); y1 = min(h, y1 + my)
    x1 = max(x1, x0 + 128); y1 = max(y1, y0 + 128)
    R[iid] = (x0, y0, x1, y1); frac.append((x1 - x0) * (y1 - y0) / (w * h))
full = sum(1 for iid in W if iid not in R or frac[list(W).index(iid)] >= 0.999) if False else None
for d in base:
    iid = d["image_id"]
    if iid not in R: out.append(d); continue
    x0, y0, x1, y1 = R[iid]; cx = d["bbox"][0] + d["bbox"][2] / 2; cy = d["bbox"][1] + d["bbox"][3] / 2
    if x0 <= cx <= x1 and y0 <= cy <= y1: out.append(d)
E = full_eval(out)
print(f"seer={sys.argv[1]} margin={m} drop={drop} AP={E.stats[0]:.4f} S/M/L={E.stats[3]:.3f}/{E.stats[4]:.3f}/{E.stats[5]:.3f} mean_area_frac={np.mean(frac):.3f} median={np.median(frac):.3f} frac>=0.95: {np.mean(np.array(frac)>=0.95):.3f}", flush=True)
