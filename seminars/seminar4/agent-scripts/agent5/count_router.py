#!/usr/bin/env python3
"""Agent 5: realistic count-based routers. (a) cascade 26s->26l escalating on 26s's own detection count (>0.25);
(b) s/l selection with 26n's detection count as the router (26n is the 'thumbnail' router proxy, cost 1.65 ms at 640)."""
import sys, numpy as np
sys.path.insert(0, "/data/tmp/ds-yolo/seminar4/work/agent5")
from oracle import MODELS, W, IMG_IDS, mix, score
N = len(IMG_IDS)
cs = np.load(f"{W}/sig_26s.npy")[:, 1]; cn = np.load(f"{W}/sig_26n.npy")[:, 1]
n = int((5.36 - 2.75) / 6.89 * N); esc = np.zeros(N, bool); esc[np.argsort(-cs, kind="stable")[:n]] = True
score(mix(np.where(esc, "26l", "26s")), f"cascade s->l on 26s count>0.25, escalate {esc.mean():.3f}, avgL={2.75+esc.mean()*6.89:.3f}")
for share, lab in [(0.630, "avgL 5.36 + router"), (0.52, "avgL 4.90 + 1.65 router = 6.55 -> for a 320 router est. 5.5")]:
    n = int(share * N); sel = np.zeros(N, bool); sel[np.argsort(-cn, kind="stable")[:n]] = True
    score(mix(np.where(sel, "26l", "26s")), f"s/l selection by 26n count>0.25, l share {sel.mean():.3f} ({lab})")
