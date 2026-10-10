"""GT-rule and oracle ceilings for a per-frame scale route on YOLO26-L, host-side engine choice (no branch penalty), measured T4 costs."""
import numpy as np, sys, json, time
from evlib import *
COST = {"l448": 4.28, "l512": 4.88, "l544": 5.32, "l576": 6.25, "l640": 6.89, "m448": 3.33, "m640": 5.34}
AP = {"l448": 0.5117, "l512": 0.5262, "l544": 0.5329, "l576": 0.5368, "l640": 0.5417, "m448": 0.4937, "m640": 0.5261}
# upper concave hull of dense (cost, AP) points, T4 unset-buffer basis
pts = sorted([(3.33,0.4937),(4.28,0.5117),(4.88,0.5262),(5.32,0.5329),(6.25,0.5368),(6.89,0.5417),(12.41,0.5691)])
hull=[]
for p in pts:
    while len(hull)>=2 and (hull[-1][1]-hull[-2][1])*(p[0]-hull[-2][0]) <= (p[1]-hull[-2][1])*(hull[-1][0]-hull[-2][0]): hull.pop()
    hull.append(p)
def env(t):
    for (x0,y0),(x1,y1) in zip(hull,hull[1:]):
        if x0<=t<=x1: return y0+(y1-y0)*(t-x0)/(x1-x0)
    return float('nan')
print("hull", hull)
names = ["l448","l512","l544","l576","l640"]
t0=time.time(); S = {n: evaluated(n) for n in names}; print("evaluated in", round(time.time()-t0), "s")
ids, cnt, small, med = gt_stats(); assert ids == S["l448"]["imgIds"]
prox = {n: per_image_proxy(S[n]) for n in names}
rng = np.random.default_rng(0)
I = len(ids)
def run(lo, hi, p_hi, label, rule):
    """route a share p_hi of images to hi by a per-image rule (higher rule value -> hi)."""
    k = int(round(p_hi*I)); order = np.argsort(-rule, kind="stable"); ch = np.zeros(I,int); ch[order[:k]] = 1
    ap = score([S[lo],S[hi]], ch)[0]; t = p_hi*COST[hi]+(1-p_hi)*COST[lo]
    print(f"{lo}/{hi} share_hi={p_hi:.2f} {label:12s} AP={ap:.4f} T4avg={t:.2f} env={env(t):.4f} margin={ap-env(t):+.4f}", flush=True)
    return ap
for lo,hi in [("l448","l640"),("l448","l544"),("l512","l640"),("l448","l512")]:
    for p in [0.25,0.5,0.75]:
        run(lo,hi,p,"random",rng.random(I))
        run(lo,hi,p,"gt_count",cnt+1e-3*rng.random(I))
        run(lo,hi,p,"gt_small",small+1e-3*cnt+1e-4*rng.random(I))
        run(lo,hi,p,"oracle_proxy",prox[hi]-prox[lo]+1e-4*rng.random(I))
