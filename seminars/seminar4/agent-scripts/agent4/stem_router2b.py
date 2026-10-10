#!/usr/bin/env python3
"""Addendum: MLP seed variance (share 0.42) and four-width share 0.34 (4.95 / 6.17 ms). Run as stem_router.py."""
import pickle, numpy as np, importlib.util, sys
import oracle as O
spec = importlib.util.spec_from_file_location("sr", "/data/tmp/ds-yolo/seminar4/work/agent4/stem_router.py")
src = open(spec.origin).read().split("# anchors")[0]           # definitions only, no main rows
ns = {}; exec(compile(src, "stem_router_defs", "exec"), ns)
d, idx, g, evs, realised_base, mlp_cv = ns["d"], ns["idx"], ns["g"], ns["evs"], ns["realised"], ns["mlp_cv"]
def realised(score, name, pl):
    K = int(round(pl * O.NI)); ch = np.array(["m"] * O.NI, dtype=object); ch[np.argsort(-score)[:K]] = "l"
    ap = O.mixed_ap(evs, list(ch))[0]
    print(f"ROUTE2 {name:30s} l_share={pl:.2f} AP={ap:.4f}", flush=True); return ap
X = d["m640"][idx]
s1 = mlp_cv(X, seed=1); s2 = mlp_cv(X, seed=2)
realised(s1, "mlp64 m640 seed1", 0.42); realised(s2, "mlp64 m640 seed2", 0.42)
realised(s1, "mlp64 m640 seed1", 0.34)
for s in range(2):
    realised(np.random.default_rng(10 + s).random(O.NI), f"random null seed {10+s}", 0.34)
realised(g, "oracle quantile", 0.34)
