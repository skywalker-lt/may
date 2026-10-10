"""Paired image bootstrap: feedback route (own448 small-count, thr 0.25) L512/L640 at p=0.3 and L448/L640 at p=0.5,
against the interpolated dense envelope (L544-L640 chord) and the share-matched random route."""
import numpy as np, sys, json, contextlib, io
from evlib import *
seed = int(sys.argv[1]); draws = int(sys.argv[2])
COST = {"l448": 4.28, "l512": 4.88, "l544": 5.32, "l640": 6.89}
S = {n: evaluated(n) for n in ["l448","l512","l544","l640"]}; ids = S["l448"]["imgIds"]; I = len(ids); pos = {i:k for k,i in enumerate(ids)}
c = np.zeros(I); cs = np.zeros(I)
for d in json.load(open(PATHS["l448"])):
    if d["score"] > 0.25:
        k = pos[d["image_id"]]; c[k] += 1; cs[k] += d["bbox"][2]*d["bbox"][3] < 32**2
sig = cs + 1e-3*c + 1e-4*np.random.default_rng(0).random(I)
E = base_eval()
def sc(srcs, ch, idx):
    St = np.stack([s["ev"] for s in srcs]); mixed = St[ch[idx], :, :, idx]  # n,C,A
    E.evalImgs = list(np.transpose(mixed, (1, 2, 0)).reshape(-1)); E.params.imgIds = [ids[i] for i in idx]; E._paramsEval = E.params
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return float(E.stats[0])
rng = np.random.default_rng(seed)
for lo, p in [("l512", 0.3), ("l448", 0.5)]:
    k = int(round(p*I)); ch = np.zeros(I, int); ch[np.argsort(-sig)[:k]] = 1
    t = p*COST["l640"] + (1-p)*COST[lo]; w = (t-5.32)/(6.89-5.32)
    for b in range(draws):
        idx = rng.integers(0, I, I) if seed >= 0 else np.arange(I)
        rr = np.zeros(I, int); rr[rng.permutation(I)[:k]] = 1
        a = sc([S[lo], S["l640"]], ch, idx); r = sc([S[lo], S["l640"]], rr, idx)
        e = (1-w)*sc([S["l544"]], np.zeros(I, int), idx) + w*sc([S["l640"]], np.zeros(I, int), idx)
        print(f"{lo}/l640 p={p} draw={b} routed={a:.4f} random={r:.4f} env={e:.4f} m_env={a-e:+.4f} m_null={a-r:+.4f}", flush=True)
