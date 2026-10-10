"""Round 3: same-backbone head routing (one-to-one vs one-to-many + CLI NMS) on public M and the two fine-tuned
checkpoints. Exact mixtures from cached evalImgs. K1', K2, K4. CPU, one thread."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
import sys, pickle, copy, contextlib, io, numpy as np
from scipy.stats import spearmanr
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
W = "/data/tmp/ds-yolo/seminar5/work/agent8/"
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO("/data/tmp/ds-yolo/seminar5/inputs/dumps/instances_val2017.json")
E = COCOeval(gt, None, "bbox"); E._paramsEval = copy.deepcopy(E.params)
P = E.params; I = len(P.imgIds); K = len(P.catIds); A = len(P.areaRng)
names = ["m640o2o_h200", "m640o2m", "spl_o2o", "spl_o2m", "ts768_o2o", "ts768_o2m"]
C = {m: pickle.load(open(W + f"cache/{m}.pkl", "rb")) for m in names}
for m in C: assert C[m]["imgIds"] == P.imgIds
S = pickle.load(open(W + "crowd_stats.pkl", "rb")); assert S["imgs"] == P.imgIds
img_of = np.tile(np.arange(I), K * A)
def ap(ev):
    E.evalImgs = ev
    with contextlib.redirect_stdout(io.StringIO()):
        E.accumulate(); E.summarize()
    return E.stats.copy()
def mix(r, a, b):
    rr = r[img_of]; return [y if t else x for x, y, t in zip(C[a]["evalImgs"], C[b]["evalImgs"], rr)]
def sub(mask, a):
    rr = mask[img_of]; return [x if t else None for x, t in zip(C[a]["evalImgs"], rr)]
# per-image AP (same definition as mix.py)
def per_image_ap(m):
    ev = C[m]["evalImgs"]; T = len(P.iouThrs); rec = np.linspace(0, 1, 101); acc = [[] for _ in range(I)]
    for k in range(K):
        base = k * A * I
        for i in range(I):
            e = ev[base + i]
            if e is None: continue
            gi = np.asarray(e["gtIgnore"]).astype(bool); npig = int((~gi).sum())
            if npig == 0: continue
            sc = np.asarray(e["dtScores"][:100]); o = np.argsort(-sc, kind="mergesort")
            dm = np.asarray(e["dtMatches"])[:, :100][:, o]; di = np.asarray(e["dtIgnore"])[:, :100][:, o]
            tp = np.cumsum(np.logical_and(dm > 0, ~di), 1).astype(float); fp = np.cumsum(np.logical_and(dm == 0, ~di), 1).astype(float)
            aps = []
            for t in range(T):
                if tp.shape[1] == 0: aps.append(0.0); continue
                rc = tp[t] / npig; pr = np.maximum.accumulate((tp[t] / (tp[t] + fp[t] + 1e-12))[::-1])[::-1]
                ids = np.searchsorted(rc, rec, side="left"); aps.append(np.where(ids < len(pr), pr[np.minimum(ids, len(pr) - 1)], 0.0).mean())
            acc[i].append(np.mean(aps))
    return np.array([np.mean(a) if a else np.nan for a in acc])
st = {m: ap(C[m]["evalImgs"]) for m in names}
print("== full val2017 (AP, AP50, AP75, AP_S, AP_M, AP_L)")
for m in names: print(f"  {m:14s}", " ".join(f"{x:.4f}" for x in st[m][:6]))
pairs = [("public M@640", "m640o2o_h200", "m640o2m"), ("splice10", "spl_o2o", "spl_o2m"), ("twoscale20@768", "ts768_o2o", "ts768_o2m")]
print("\n== head offset o2m - o2o (K4)")
for lab, a, b in pairs:
    d = st[b][:6] - st[a][:6]; print(f"  {lab:15s} AP {d[0]:+.4f} AP50 {d[1]:+.4f} AP75 {d[2]:+.4f} S {d[3]:+.4f} M {d[4]:+.4f} L {d[5]:+.4f}")
n, c25, sp, p3 = S["n"].astype(float), S["c25"].astype(float), S["sp"].astype(float), S["p3"].astype(float)
pn = S["preds"][("n320", "log1p GT count")]
print("\n== subset AP by GT count bin, o2m - o2o")
bins = [(1, 2), (3, 5), (6, 10), (11, 10**6)]
for lab, a, b in pairs:
    row = []
    for lo, hi in bins:
        mk = (n >= lo) & (n <= hi); row.append(f"{lo}-{hi if hi < 1e5 else '+'} ({int(mk.sum())}): {ap(sub(mk, b))[0] - ap(sub(mk, a))[0]:+.4f}")
    print(f"  {lab:15s}", " | ".join(row))
g = {}
for lab, a, b in pairs:
    g[lab] = per_image_ap(b) - per_image_ap(a)
ok = ~np.isnan(g["public M@640"])
gm = g["public M@640"]
print(f"\n== per-image gain public M: mean {np.nanmean(gm):+.4f}, sd {np.nanstd(gm):.4f}, better {np.mean(gm[ok] > 0):.3f}, worse {np.mean(gm[ok] < 0):.3f}, equal {np.mean(gm[ok] == 0):.3f}")
sigs = {"GT count": n, "GT pairs>0.3": p3 + 1e-3 * n, "o2o own count": c25 + 1e-3 * sp, "n320 ridge count": pn}
for k, v in sigs.items(): print(f"  Spearman({k}, gain) = {spearmanr(v[ok], gm[ok])[0]:+.3f}")
for lab in ("splice10", "twoscale20@768"):
    o2 = ok & ~np.isnan(g[lab]); print(f"  Spearman(per-image gain public M, {lab}) = {spearmanr(gm[o2], g[lab][o2])[0]:+.3f}")
# split-half stability of the per-image gain sign is not available (one dump per head); cross-checkpoint rho above is the stability test
rng = np.random.default_rng(8)
base, alt = "m640o2o_h200", "m640o2m"; a0 = st[base][0]; a1 = st[alt][0]
print(f"\n== routes on public M: o2o {a0:.4f}, o2m everywhere {a1:.4f}")
gg = np.where(ok, gm, -1e9)
for share in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    q = int(round(share * I)); out = []
    r = np.zeros(I, bool); r[np.argsort(-gg, kind="mergesort")[:q]] = True; orc = ap(mix(r, base, alt))[0]
    for k, v in sigs.items():
        jit = 1e-6 * rng.random(I)
        r = np.zeros(I, bool); r[np.argsort(v + jit, kind="mergesort")[:q]] = True; lo = ap(mix(r, base, alt))[0]
        r = np.zeros(I, bool); r[np.argsort(-(v + jit), kind="mergesort")[:q]] = True; hi = ap(mix(r, base, alt))[0]
        out.append((k, lo, hi))
    nl = []
    for d in range(5):
        r = np.zeros(I, bool); r[rng.choice(I, q, replace=False)] = True; nl.append(ap(mix(r, base, alt))[0])
    nm, ns = np.mean(nl), np.std(nl)
    print(f"  share {share:.1f}: oracle {orc:.4f} | null {nm:.4f} (sd {ns:.4f}, linear {a0 + share * (a1 - a0):.4f})")
    for k, lo, hi in out:
        print(f"      {k:17s} sparse->o2m {lo:.4f} ({lo - nm:+.4f} vs null, {lo - a0:+.4f} vs o2o) | busy->o2m {hi:.4f} ({hi - nm:+.4f}, {hi - a0:+.4f})")
