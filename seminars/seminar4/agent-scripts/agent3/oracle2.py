#!/usr/bin/env python3
"""Agent 3, second pass: exact (Lagrangian) oracle under a mean-latency budget, plus realisable and label-oracle
routes. Per-image choice c_i = argmax_k (q_k[i] - lam * L_k); lam bisected so the mean latency hits the budget.
Realisable routes use only a scalar of the yolo26s output (no ground truth) with the oracle's shares; label-oracle
routes use the ground-truth object count or small-object count of the image (what a thumbnail router could at best
learn), again with the oracle's shares. Same q proxy as oracle.py (TAU 0.25)."""
import json, os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "2"
sys.argv = [sys.argv[0], "0.25"]
exec(open("/data/tmp/ds-yolo/seminar4/work/agent3/oracle.py").read().split("print(f\"\\nTAU")[0])  # reuse loading + q
np.savez(f"{OUT}/q_tau0.25.npz", **{k: q[k] for k in q}, img_ids=np.array(img_ids))
nI = len(img_ids)


def lagr(models, budget):
    Q = np.stack([q[k] for k in models], 1); L = np.array([LAT[k] for k in models])
    lo, hi = 0.0, 1.0
    for _ in range(60):
        lam = (lo + hi) / 2; c = np.argmax(Q - lam * L[None, :], 1)
        if L[c].mean() > budget: lo = lam
        else: hi = lam
    c = np.argmax(Q - hi * L[None, :], 1)
    return [models[j] for j in c], L[c].mean()


def shares_of(choice, models): return {k: round(choice.count(k) / nI, 3) for k in models}


def route_by_signal(sig, models, shares):
    """Assign heavier models to images with larger signal, with the given shares (quantile bins)."""
    order = np.argsort(-np.asarray(sig)); choice = [None] * nI; pos = 0
    for k in sorted(models, key=lambda k: -LAT[k]):
        n = int(round(shares[k] * nI))
        for j in order[pos:pos + n]: choice[j] = k
        pos += n
    for j in range(nI):
        if choice[j] is None: choice[j] = min(models, key=lambda k: LAT[k])
    return choice


# ground-truth signals and s-output signals
ngt = np.zeros(nI); nsmall = np.zeros(nI); s_unc = np.zeros(nI); s_cnt = np.zeros(nI); s_negmax = np.zeros(nI)
for j, i in enumerate(img_ids):
    anns = gt.loadAnns(gt.getAnnIds(imgIds=i, iscrowd=False)); ngt[j] = len(anns)
    nsmall[j] = sum(1 for a in anns if a["area"] < 32 * 32)
    ds = sorted([d["score"] for d in by_img["s"].get(i, [])], reverse=True)
    s_unc[j] = sum(1 for x in ds if 0.1 <= x < 0.5); s_cnt[j] = sum(1 for x in ds if x >= 0.25); s_negmax[j] = -(ds[0] if ds else 0)

print("\nExact oracle (Lagrangian), TAU 0.25:")
res = {}
for name, models, budget in [("n+x @5.36", "nx", 5.36), ("n+m+x @5.36", "nmx", 5.36), ("s+m+l+x @5.36", "smlx", 5.36),
                             ("n..x @5.36", "nsmlx", 5.36), ("n..x @2.75", "nsmlx", 2.75), ("n..x @6.89", "nsmlx", 6.89),
                             ("s+l @5.36", "sl", 5.36), ("m+l+x @6.89", "mlx", 6.89)]:
    choice, avg = lagr(list(models), budget); a = score_mix(choice); sh = shares_of(choice, models)
    res[name] = (avg, a, sh); print(f"{name:16s} avg {avg:.2f} ms AP {a:.4f} shares {sh}", flush=True)

print("\nRoutes with the s+m+l+x @5.36 oracle shares, by a signal (no ground-truth choice):")
sh = res["s+m+l+x @5.36"][2]
for sname, sig in [("GT object count", ngt), ("GT small-object count", nsmall), ("s: boxes with 0.1<=score<0.5", s_unc),
                   ("s: boxes with score>=0.25", s_cnt), ("s: negative max score", s_negmax)]:
    choice = route_by_signal(sig, "smlx", sh); avg = np.mean([LAT[k] for k in choice]); a = score_mix(choice)
    print(f"  {sname:32s} avg {avg:.2f} ms AP {a:.4f}", flush=True)
# agreement between the oracle choice and the GT-count route
oc, _ = lagr(list("smlx"), 5.36); rc = route_by_signal(ngt, "smlx", sh)
print("  oracle/GT-count route agreement:", round(np.mean([a == b for a, b in zip(oc, rc)]), 3))
json.dump({k: [v[0], v[1], v[2]] for k, v in res.items()}, open(f"{OUT}/oracle2.json", "w"), indent=1)
