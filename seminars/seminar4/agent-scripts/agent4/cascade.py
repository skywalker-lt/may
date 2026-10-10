#!/usr/bin/env python3
"""Candidate B receipt: does a scalar of YOLO26-n's output predict where the heavy model helps?
Gate scalars from n's dump per image: max score, number of boxes with score>0.25, mean of top-5 scores.
Policy: run the heavy model (m or l) on the images with the LOWEST gate value (least confident), with the share fixed
so that the average latency (n always + heavy on share p) equals 5.36 ms. Reported AP is exact mixed pycocotools AP.
Also: Spearman correlation between each scalar and the per-image proxy gain (heavy - n), and the same policy driven
by the ORACLE gain (upper bound for any gate at that share)."""
import json, pickle, numpy as np
from scipy.stats import spearmanr
import oracle as O

evs, proxy = pickle.load(open(f"{O.W}/evals.pkl", "rb"))
keys = list(O.MODELS)
P = np.load(f"{O.W}/proxy.npy")
ki = {k: j for j, k in enumerate(keys)}
dets = json.load(open(O.MODELS["n"]))
by_img = {}
for d in dets:
    by_img.setdefault(d["image_id"], []).append(d["score"])
mx, cnt, top5 = np.zeros(O.NI), np.zeros(O.NI), np.zeros(O.NI)
for i, iid in enumerate(O.imgIds):
    s = np.sort(by_img.get(iid, [0.0]))[::-1]
    mx[i], cnt[i], top5[i] = s[0], (s > 0.25).sum(), s[:5].mean()
print("gate scalars done", flush=True)
for heavy, p in [("m", (5.36 - 1.65) / 5.36), ("l", (5.36 - 1.65) / 6.89)]:
    gain = P[:, ki[heavy]] - P[:, ki["n"]]
    nh = int(round(p * O.NI))
    print(f"--- heavy={heavy} share={p:.3f} (n always; avg latency 5.36)")
    for name, g, sign in [("max_score", mx, 1), ("count>0.25", cnt, -1), ("top5_mean", top5, 1), ("ORACLE gain", gain, -1)]:
        rho = spearmanr(g, gain).correlation
        order = np.argsort(sign * g)  # ascending in sign*g: least confident first (or largest gain first)
        ch = ["n"] * O.NI
        for i in order[:nh]:
            ch[i] = heavy
        ap, aps, apm, apl = O.mixed_ap(evs, ch)
        print(f"CASCADE heavy={heavy} gate={name} spearman(gate,gain)={rho:+.3f} AP={ap:.4f} S/M/L={aps:.3f}/{apm:.3f}/{apl:.3f} margin_vs_M={ap-0.5261:+.4f}", flush=True)
    # random gate at the same share
    rng = np.random.default_rng(0)
    ch = ["n"] * O.NI
    for i in rng.permutation(O.NI)[:nh]:
        ch[i] = heavy
    ap = O.mixed_ap(evs, ch)[0]
    print(f"CASCADE heavy={heavy} gate=random AP={ap:.4f} margin_vs_M={ap-0.5261:+.4f}", flush=True)
