#!/usr/bin/env python3
"""Round-3: two-scale C (M at 512 / M at 768) re-budgeted with MI3's engine-level costs.
768 branch 7.48 ms unset (MI3), 512 branch est. 3.80 + 0.65 = 4.45 ms (pending round 4); at a 5.36 ms average the
768 share is 0.30. Rows: oracle quantile (true per-image gain 768 - 512), random null, GT small-object-count rule
(A2's rule: most GT objects under 32x32 px -> 768). Exact global pycocotools AP of each mixture on full val2017.
Run: OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python res_budget.py > res_budget.log
"""
import pickle, numpy as np
import oracle as O
W = O.W
e512, p512 = pickle.load(open(f"{W}/eval_m512.pkl", "rb"))
e768, p768 = pickle.load(open(f"{W}/eval_m768.pkl", "rb"))
evs = {"a": e512, "b": e768}
g = np.asarray(p768) - np.asarray(p512)
small = np.array([sum(1 for a in O.gt.loadAnns(O.gt.getAnnIds(imgIds=[i], iscrowd=None)) if a["area"] < 32 * 32)
                  for i in O.imgIds], float)
for share in (0.30, 0.44):
    K = int(round(share * O.NI))
    ms = (1 - share) * 4.45 + share * 7.48
    for name, score in (("oracle quantile", g), ("random null", np.random.default_rng(0).random(O.NI)),
                        ("GT small-count rule", small + 1e-3 * np.random.default_rng(1).random(O.NI))):
        ch = np.array(["a"] * O.NI, dtype=object); ch[np.argsort(-score)[:K]] = "b"
        ap, aps, apm, apl = O.mixed_ap(evs, list(ch))
        print(f"RES share768={share:.2f} avg_ms_est={ms:.2f} {name:20s} AP={ap:.4f} S/M/L={aps:.3f}/{apm:.3f}/{apl:.3f}", flush=True)
