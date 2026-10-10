"""Round 2 self-check: agent 7 reports that YOLOv12-M routed by LOW predicted count at share 0.5 reaches 0.5262 vs null
0.5203. Route sparse images (low count) from the o2o yolo26m to the NMS proxies, shares 0.2-0.6, GT-free signals."""
import os, sys, pickle, copy, contextlib, io, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
W = "/data/tmp/ds-yolo/seminar5/work/agent8/"; D = "/data/tmp/ds-yolo/seminar5/inputs/dumps/"
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(D + "instances_val2017.json")
E = COCOeval(gt, None, "bbox"); E._paramsEval = copy.deepcopy(E.params)
P = E.params; I = len(P.imgIds); K = len(P.catIds); A = len(P.areaRng)
C = {m: pickle.load(open(W + f"cache/{m}.pkl", "rb")) for m in ("m26", "v12m", "m11")}
S = pickle.load(open(W + "crowd_stats.pkl", "rb")); img_of = np.tile(np.arange(I), K * A)
def ap(ev):
    E.evalImgs = ev
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate(); E.summarize()
    return E.stats[0]
def mix(r, a, b):
    rr = r[img_of]; return [y if t else x for x, y, t in zip(C[a]["evalImgs"], C[b]["evalImgs"], rr)]
n, c25 = S["n"].astype(float), S["c25"].astype(float); pn = S["preds"][("n320", "log1p GT count")]
sig = {"GT count (bound)": n, "o2o own count>=0.25": c25 + 1e-3 * n * 0, "n320 ridge count": pn}
rng = np.random.default_rng(5)
for alt in ("v12m", "m11"):
    print(f"== base m26 {ap(C['m26']['evalImgs']):.4f}, alt {alt} {ap(C[alt]['evalImgs']):.4f}", flush=True)
    for share in (0.2, 0.4, 0.5, 0.6):
        q = int(round(share * I)); out = []
        for k, v in sig.items():
            r = np.zeros(I, bool); r[np.argsort(v + 1e-6 * rng.random(I), kind="mergesort")[:q]] = True
            out.append(f"{k} {ap(mix(r, 'm26', alt)):.4f}")
        nl = []
        for d in range(3):
            r = np.zeros(I, bool); r[rng.choice(I, q, replace=False)] = True; nl.append(ap(mix(r, "m26", alt)))
        print(f"  share {share:.1f} low-count->{alt}: " + " | ".join(out) + f" | null {np.mean(nl):.4f} (sd {np.std(nl):.4f})", flush=True)
