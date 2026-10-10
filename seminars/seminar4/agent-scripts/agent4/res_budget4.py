#!/usr/bin/env python3
"""Round 4 (agent 4): C (M at 512 / M at 768) re-scored at the share the last T4 receipts allow (round4/moderator_insert.md
section 3: 512 branch 3.47, 768 branch 7.115, plain M 5.05 real-input -> 768 share 0.43 at M's own latency; 0.52 at the 5.36 unit).
Rows: oracle quantile, random null (two seeds), GT small-object-count rule. Exact global pycocotools AP, full val2017.
Run: OMP_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python res_budget4.py > res_budget4.log
"""
import pickle, numpy as np, sys
sys.path.insert(0, "/data/tmp/ds-yolo/seminar4/work/agent4")
import oracle as O
W = O.W
e512, p512 = pickle.load(open(f"{W}/eval_m512.pkl", "rb")); e768, p768 = pickle.load(open(f"{W}/eval_m768.pkl", "rb"))
evs = {"a": e512, "b": e768}; g = np.asarray(p768) - np.asarray(p512)
small = np.array([sum(1 for a in O.gt.loadAnns(O.gt.getAnnIds(imgIds=[i], iscrowd=None)) if a["area"] < 32 * 32) for i in O.imgIds], float)
for share in (0.43, 0.52):
    K = int(round(share * O.NI)); ms = (1 - share) * 3.47 + share * 7.115
    for name, score in (("oracle quantile", g), ("random null s0", np.random.default_rng(0).random(O.NI)), ("random null s1", np.random.default_rng(5).random(O.NI)),
                        ("GT small-count rule", small + 1e-3 * np.random.default_rng(1).random(O.NI))):
        ch = np.array(["a"] * O.NI, dtype=object); ch[np.argsort(-score)[:K]] = "b"
        ap, aps, apm, apl = O.mixed_ap(evs, list(ch))
        print(f"RES share768={share:.2f} avg_ms_real={ms:.2f} {name:20s} AP={ap:.4f} S/M/L={aps:.3f}/{apm:.3f}/{apl:.3f}", flush=True)
