"""Round 3: how much router quality the L ladder needs to clear the envelope + 0.003. Routers: thumbnail ridge (n320, Spearman 0.649),
YOLO26-N@640's own detection count (a stand-in for a trained count head; N is a full pass, cost noted), GT count + noise at set Spearman,
GT count. Cost model A, T4 unset est., target 5.36 ms, shares fixed as in envelope.py (so router cost is held at 0.18 ms)."""
import os, sys, json, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7")
from scipy.stats import spearmanr
from headroom import I, cnt_pred_all, cnt_pred_small, n_all
from mixlib import mix_ap, img_ids
ids = img_ids(); pos = {i: k for k, i in enumerate(ids)}
det = json.load(open("/data/tmp/ds-yolo/seminar5/inputs/dumps/dump_yolo26n_coco.json"))
for tau in (0.25, 0.35):
    c = np.zeros(I)
    for d in det:
        if d["score"] >= tau: c[pos[d["image_id"]]] += 1
    print(f"YOLO26-N@640 own count (score>={tau}): Spearman with GT count {spearmanr(c, n_all)[0]:.3f}", flush=True)
    if tau == 0.25: ncnt = c
print(f"thumbnail ridge: Spearman {spearmanr(cnt_pred_all, n_all)[0]:.3f}", flush=True)
def ranks(score): r = np.empty(I, int); r[np.argsort(score + 1e-9 * np.arange(I))] = np.arange(I); return r
def assign(r, sh): return np.searchsorted(np.cumsum(sh)[:-1] * I, r, side="right")
MEN = {"L@448/L@512/L@640": (["dumpml_yolo26l_448_coco", "dumpml_yolo26l_512_coco", "dump_yolo26l_coco"], [0.33, 0.374, 0.296]),
       "M@448/L@512/L@640": (["dumpml_yolo26m_448_coco", "dumpml_yolo26l_512_coco", "dump_yolo26l_coco"], [0.2, 0.471, 0.329])}
rng = np.random.RandomState(7); g = np.log1p(n_all); z = rng.randn(I)
routers = {"YOLO26-N own count": ncnt + 1e-3 * rng.rand(I)}
for sig in (0.55, 0.35):
    s = g + sig * z; routers[f"GT+noise sd {sig} (Spearman {spearmanr(s, n_all)[0]:.3f})"] = s
for m, (names, sh) in MEN.items():
    for k, s in routers.items():
        print(f"{m:20s} router {k:34s} AP {mix_ap(names, assign(ranks(s), sh)):.4f}  (envelope + 0.003 = 0.5347)", flush=True)
