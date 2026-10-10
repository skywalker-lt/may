"""Run pycocotools evaluate() once per dump, save evalImgs (for exact mixtures via accumulate) and per-image proxy AP."""
import os, sys, json, pickle, time, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io
D = "/data/tmp/ds-yolo/seminar5/inputs/dumps"; W = "/data/tmp/ds-yolo/seminar5/work/agent7/ev"
os.makedirs(W, exist_ok=True)
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(f"{D}/instances_val2017.json")
names = sys.argv[1:]
for nm in names:
    if os.path.exists(f"{W}/{nm}.pkl"): continue
    t0 = time.time()
    dts = json.load(open(f"{D}/{nm}.json"))
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(dts)
        E = COCOeval(gt, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    imgIds = list(E.params.imgIds); catIds = list(E.params.catIds)
    # per-image proxy AP (area all, maxDet 100), mean over GT categories and 10 IoU thresholds
    I, A = len(imgIds), len(E.params.areaRng); rec = np.linspace(0, 1, 101)
    proxy = np.zeros(I); ncat = np.zeros(I)
    for k in range(len(catIds)):
        for i in range(I):
            e = E.evalImgs[k * A * I + 0 * I + i]
            if e is None: continue
            gtIg = e["gtIgnore"].astype(bool); G = int((~gtIg).sum())
            if G == 0: continue
            dm = e["dtMatches"][:, :100]; dI = e["dtIgnore"][:, :100].astype(bool)
            aps = []
            for t in range(dm.shape[0]):
                keep = ~dI[t]; tp = (dm[t][keep] > 0).astype(float); fp = 1 - tp
                if tp.size == 0: aps.append(0.0); continue
                ctp, cfp = np.cumsum(tp), np.cumsum(fp); r = ctp / G; p = ctp / (ctp + cfp)
                p = np.maximum.accumulate(p[::-1])[::-1]
                idx = np.searchsorted(r, rec, side="left"); q = np.where(idx < len(p), p[np.minimum(idx, len(p) - 1)], 0)
                aps.append(q.mean())
            proxy[i] += np.mean(aps); ncat[i] += 1
    proxy = np.where(ncat > 0, proxy / np.maximum(ncat, 1), np.nan)
    pickle.dump({"evalImgs": E.evalImgs, "imgIds": imgIds, "stats": E.stats, "proxy": proxy}, open(f"{W}/{nm}.pkl", "wb"), protocol=4)
    print(f"{nm} AP={E.stats[0]:.4f} APs/m/l={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f} ndt={len(dts)} {time.time()-t0:.0f}s", flush=True)
