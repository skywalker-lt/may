"""Seed spread of the best cross-fitted partitions, and the one-weight-set (L only) restriction."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent9/r3")
import scene_envelope as S
full_keys, full_cost, full_P, full_names = list(S.keys), S.cost.copy(), S.P.copy(), list(S.names)
def restrict(fam):
    j = [i for i, k in enumerate(full_keys) if fam is None or k.startswith(fam)]
    S.keys = [full_keys[i] for i in j]; S.cost = full_cost[j]; S.P = full_P[:, j]; S.names = [full_names[i] for i in j]
for target in (4.70, 5.36):
    print(f"== budget {target}", flush=True)
    restrict(None)
    for part, K in (("m640", 4), ("predcount", 4)):
        for seed in (1, 2): S.report(f"{part} K={K} seed {seed}", S.crossfit(part, K, target, seed))
    restrict("L")
    for part, K in (("predcount", 4), ("m640", 4), ("GTcount", 4)): S.report(f"[L only] {part} K={K}", S.crossfit(part, K, target, 0))
