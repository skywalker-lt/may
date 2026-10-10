"""Round 4, agent 3. (a) The L4 front from the L4 table (inputs/receipts_l4/l4.log, moderator insert) and the bar
every L4 design meets at its latency; (b) the two-scale 512/768 mixture at the 768 share the last T4 receipt allows
(512 branch 3.45-3.49 real, 768 branch 7.10-7.13 real, M 5.04-5.06 real): random null, GT small-count rule, GT median
area rule, and ridge routers (5-fold out of fold) on the pooled stem features; (c) the M/L two-tail design priced on
the L4 at its own latency.  CPU, 2 threads.
Command: cd work/agent3 && OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python l4_check.py
"""
import json, os, sys, numpy as np, contextlib, io
os.environ.setdefault("OMP_NUM_THREADS", "2")
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ---------------- (a) the L4 front ----------------
L4 = {"n": 0.917, "s": 1.344, "m": 2.551, "l": 3.346, "x": 6.204}
T4 = {"n": 1.65, "s": 2.75, "m": 5.37, "l": 6.89, "x": 12.41}
AP = {"n": 0.4060, "s": 0.4795, "m": 0.5261, "l": 0.5417, "x": 0.5691}
slope_l4 = (AP["l"] - AP["m"]) / (L4["l"] - L4["m"]); slope_t4 = (AP["l"] - AP["m"]) / (T4["l"] - T4["m"])
print(f"L4/T4 ratios: " + ", ".join(f"{k} {L4[k]/T4[k]:.3f}" for k in L4))
print(f"m-l chord slope: T4 {slope_t4:.5f} AP/ms, L4 {slope_l4:.5f} AP/ms (ratio {slope_l4/slope_t4:.2f})")
F4 = lambda t: AP["m"] + slope_l4 * (t - L4["m"])
# log-linear fit over n/s/m/l/x for reference
x = np.log(np.array([L4[k] for k in "nsmlx"])); y = np.array([AP[k] for k in "nsmlx"]); a, b = np.polyfit(x, y, 1)
print(f"log fit over n..x on the L4: AP = {b:.4f} + {a:.4f} ln(ms); residuals " + ", ".join(f"{k} {AP[k]-(b+a*np.log(L4[k])):+.4f}" for k in "nsmlx"))
rows = [("M", 2.551), ("M real", 2.42), ("M+0.45", 3.0), ("L", 3.346), ("M at 768", 3.642), ("ML at share 0.16 (ratio-priced)", 2.635 + 0.16 * 0.58),
        ("ML at share 0.31", 2.635 + 0.31 * 0.58), ("soft mixture, 12 neck 1x1 at P4 cost", 2.551 + 12 * 0.118), ("soft mixture, 12 at P3 cost", 2.551 + 12 * 0.175),
        ("EsMoE-M sdpa", 4.773), ("crop pass est. 1.0-1.3 ms", 3.65), ("full attention x2 at P4 added", 2.551 + 0.405), ("X", 6.204)]
for name, t in rows:
    print(f"  bar on the L4 at {t:.3f} ms ({name}): front {F4(t):.4f}, bar {F4(t)+0.003:.4f}")
# pixel scaling on both devices
for dev, m512, m640, m768 in (("T4", 3.78, 5.37, 6.96), ("L4", 2.046, 2.551, 3.642)):
    print(f"{dev}: 512/640 {m512/m640:.3f} (pixels 0.64), 768/640 {m768/m640:.3f} (pixels 1.44); 640->768 difference {m768-m640:.2f} ms")
# soft-mixture micro graph: where the per-layer cost goes (profile_l4micro_softmoe4_p3.txt, per layer of 8)
p3 = dict(expert_conv=0.052, reducesum=0.0441, mul=0.035, reformat_in=0.0266, reformat_out=0.0149, gate_conv=0.0083, silu=0.0082, softmax=0.0052 + 0.004)
print("soft mixture at P3 per layer (profile): " + ", ".join(f"{k} {v:.3f}" for k, v in p3.items()) + f"; sum {sum(p3.values()):.3f}; dense 0.021; glue share {(sum(p3.values())-p3['expert_conv'])/sum(p3.values()):.2f}")
sys.stdout.flush()

# ---------------- (b) two-scale C at the new share ----------------
D = "/data/tmp/ds-yolo/seminar4/inputs/dumps"
gt = COCO(f"{D}/instances_val2017.json"); img_ids = sorted(gt.getImgIds()); nI = len(img_ids); idx = {i: k for k, i in enumerate(img_ids)}
rng = np.random.default_rng(0)


def run_eval(dets):
    dt = gt.loadRes(dets); ev = COCOeval(gt, dt, "bbox"); ev.params.imgIds = img_ids
    with contextlib.redirect_stdout(io.StringIO()):
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev


by_img, APm = {}, {}
for k, f in [("512", "dumpml_yolo26m_512_coco.json"), ("768", "dumpml_yolo26m_768_coco.json")]:
    dets = json.load(open(f"{D}/{f}")); APm[k] = run_eval(dets).stats[0]
    d = {}
    for x in dets: d.setdefault(x["image_id"], []).append(x)
    by_img[k] = d
    print(f"M at {k}: AP {APm[k]:.4f}", flush=True)
ngt = np.zeros(nI); nsm = np.zeros(nI); areas = [[] for _ in range(nI)]
for a in gt.loadAnns(gt.getAnnIds(imgIds=img_ids)):
    if a.get("iscrowd", 0): continue
    j = idx[a["image_id"]]; ngt[j] += 1; areas[j].append(a["area"])
    if a["area"] < 32 ** 2: nsm[j] += 1
logmed = np.array([np.log(np.median(v)) if v else np.log(640 * 640) for v in areas])

z = np.load(f"{D}/val2017_stem_pooled.npz"); order = np.argsort(z["image_id"]); sel = order[np.searchsorted(z["image_id"][order], img_ids)]
feats = {"m640": z["m640"][sel].astype(np.float64), "n320": z["n320"][sel].astype(np.float64)}


def ridge_oof(X, y, alpha, folds=5):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); X = np.hstack([X, np.ones((len(X), 1))])
    perm = rng.permutation(len(y)); out = np.zeros(len(y))
    for f in range(folds):
        te = perm[f::folds]; tr = np.setdiff1d(perm, te)
        A = X[tr].T @ X[tr] + alpha * np.eye(X.shape[1]); w = np.linalg.solve(A, X[tr].T @ y[tr]); out[te] = X[te] @ w
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])


def score(choice): return run_eval([d for i, k in zip(img_ids, choice) for d in by_img[k].get(i, [])]).stats[0]


def route(sig, p):  # top p by sig -> 768
    n = int(round(p * nI)); ch = ["512"] * nI
    for j in np.argsort(-sig)[:n]: ch[j] = "768"
    return ch


targets = {"nsmall": np.log1p(nsm), "neg_logmed": -logmed}
preds = {}
for fname, X in feats.items():
    for tname, yv in targets.items():
        best = None
        for alpha in (10.0, 100.0, 1000.0):
            p = ridge_oof(X, yv, alpha); r = spearman(p, yv)
            if best is None or r > best[0]: best = (r, alpha, p)
        print(f"ridge {fname}->{tname}: best alpha {best[1]:g}, OOF Spearman {best[0]:+.3f}", flush=True)
        preds[f"{fname}->{tname}"] = best[2]

# shares: 0.43 (T4 real-input pricing: 512 branch 3.47, 768 branch 7.115, M 5.05) and 0.30 (the round-3 estimate) for reference
for p in (0.43, 0.30):
    print(f"\n== 768 share {p}: arithmetic null {APm['512'] + p * (APm['768'] - APm['512']):.4f}", flush=True)
    print(f"random route: {score(list(rng.permutation(route(rng.random(nI), p)))):.4f}", flush=True)
    print(f"GT small-count rule: {score(route(nsm + 1e-3 * rng.random(nI), p)):.4f}", flush=True)
    print(f"GT smallest-median-area rule: {score(route(-logmed + 1e-6 * rng.random(nI), p)):.4f}", flush=True)
    for name in ("m640->nsmall", "m640->neg_logmed", "n320->nsmall", "n320->neg_logmed"):
        print(f"learned router {name}: {score(route(preds[name], p)):.4f}", flush=True)
print("DONE")
