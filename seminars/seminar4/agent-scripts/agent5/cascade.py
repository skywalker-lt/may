#!/usr/bin/env python3
"""Agent 5: cascade (candidate B) analysis from cached per-image proxies and cheap-model confidence signals.

Escalation costs L_cheap + L_heavy (both run). Oracle: escalate exactly where the heavy proxy beats the cheap one,
cheapest subset first by gain, until the budget. Realistic: escalate on the cheap model's own signal (max score,
count above 0.25, mean of top-5), threshold chosen to meet the budget. Reports real pycocotools AP of each mixture.
"""
import sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/data/tmp/ds-yolo/seminar4/work/agent5")
from oracle import MODELS, W, IMG_IDS, mix, score

cheap, heavy, budget = sys.argv[1], sys.argv[2], float(sys.argv[3])
Pc, Ph = np.load(f"{W}/pi_{cheap}.npy"), np.load(f"{W}/pi_{heavy}.npy")
Lc, Lh = MODELS[cheap][1], MODELS[heavy][1]
sig = np.load(f"{W}/sig_{cheap}.npy")  # columns: max score, n>0.25, mean top-5
gain = Ph - Pc
pmax = (budget - Lc) / Lh  # share of images that may escalate
print(f"{cheap}->{heavy}: budget {budget} ms allows escalation on {pmax:.1%} of images; "
      f"heavy better on {np.mean(gain > 0):.1%}, cheap better on {np.mean(gain < 0):.1%}, equal {np.mean(gain == 0):.1%}")
for j, nm in enumerate(["max score", "n>0.25", "mean top-5"]):
    rho = spearmanr(sig[:, j], gain).correlation
    print(f"  spearman({nm}, heavy-cheap gain) = {rho:+.3f}")
n = int(pmax * len(IMG_IDS))
# oracle: escalate the n images with the largest gain
esc = np.zeros(len(IMG_IDS), bool); esc[np.argsort(-gain)[:n]] = True; esc &= gain > 0
ch = np.where(esc, heavy, cheap); avgL = Lc + esc.mean() * Lh
score(mix(list(ch)), f"  oracle cascade {cheap}->{heavy}: escalate {esc.mean():.1%}, avgL={avgL:.3f}, worst={Lc+Lh:.2f}")
# realistic: escalate where the cheap model's max score is lowest (least confident)
for j, nm in enumerate(["max score", "mean top-5"]):
    esc = np.zeros(len(IMG_IDS), bool); esc[np.argsort(sig[:, j])[:n]] = True
    ch = np.where(esc, heavy, cheap); avgL = Lc + esc.mean() * Lh
    score(mix(list(ch)), f"  signal '{nm}' cascade {cheap}->{heavy}: escalate {esc.mean():.1%}, avgL={avgL:.3f}")
