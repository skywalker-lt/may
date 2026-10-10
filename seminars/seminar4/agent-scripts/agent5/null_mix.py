#!/usr/bin/env python3
"""Agent 5: random-assignment null at given shares (what a router with zero information gets), same latency."""
import sys
import numpy as np
sys.path.insert(0, "/data/tmp/ds-yolo/seminar4/work/agent5")
from oracle import MODELS, IMG_IDS, mix, score

names = sys.argv[1].split(","); shares = np.array([float(x) for x in sys.argv[2].split(",")]); seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
rng = np.random.default_rng(seed)
c = rng.choice(len(names), size=len(IMG_IDS), p=shares / shares.sum())
L = np.array([MODELS[n][1] for n in names])
score(mix([names[i] for i in c]), f"random null seed {seed} shares={dict(zip(names, shares))} avgL={L[c].mean():.3f}")
