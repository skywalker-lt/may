#!/usr/bin/env python3
"""Agent 5: how predictable is 'the heavy model wins' from image statistics a router could plausibly see?
Rule routers on ground-truth object count / size (an upper bound on a router that estimates them), s/l tails,
budget 5.36 ms average. Also the random null at the five-model oracle shares."""
import sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/data/tmp/ds-yolo/seminar4/work/agent5")
from oracle import MODELS, W, IMG_IDS, gt, mix, score

P = {n: np.load(f"{W}/pi_{n}.npy") for n in ["26n", "26s", "26m", "26l", "26x"]}
cnt = np.array([len(gt.getAnnIds(imgIds=i, iscrowd=False)) for i in IMG_IDS])
med = np.array([np.median([a["area"] for a in gt.loadAnns(gt.getAnnIds(imgIds=i, iscrowd=False))] or [0]) for i in IMG_IDS])
g = P["26l"] - P["26s"]
print(f"s->l gain: spearman(count)={spearmanr(cnt, g).correlation:+.3f} spearman(median area)={spearmanr(med, g).correlation:+.3f} "
      f"l better {np.mean(g>0):.1%} s better {np.mean(g<0):.1%}")
g2 = P["26l"] - P["26m"]
print(f"m->l gain: spearman(count)={spearmanr(cnt, g2).correlation:+.3f} spearman(median area)={spearmanr(med, g2).correlation:+.3f} "
      f"l better {np.mean(g2>0):.1%} m better {np.mean(g2<0):.1%}")
Ls, Ll = MODELS["26s"][1], MODELS["26l"][1]
pl = (5.36 - Ls) / (Ll - Ls)  # share of l that averages 5.36
n_l = int(pl * len(IMG_IDS))
print(f"s/l tails: l share for 5.36 ms average = {pl:.3f}")
# oracle s/l
sel = np.zeros(len(IMG_IDS), bool); sel[np.argsort(-g)[:n_l]] = True
score(mix(np.where(sel, "26l", "26s")), f"  s/l oracle, l share {sel.mean():.3f}")
# rule: l on the images with the most objects
sel = np.zeros(len(IMG_IDS), bool); sel[np.argsort(-cnt, kind="stable")[:n_l]] = True
score(mix(np.where(sel, "26l", "26s")), f"  s/l rule 'most objects -> l', l share {sel.mean():.3f}")
# rule: l on the images with the smallest median object
sel = np.zeros(len(IMG_IDS), bool); sel[np.argsort(med, kind="stable")[:n_l]] = True
score(mix(np.where(sel, "26l", "26s")), f"  s/l rule 'smallest objects -> l', l share {sel.mean():.3f}")
# random null, same share
rng = np.random.default_rng(0); sel = rng.random(len(IMG_IDS)) < pl
score(mix(np.where(sel, "26l", "26s")), f"  s/l random null, l share {sel.mean():.3f}")
# five-model random null at the oracle's 5.36 ms shares
names = ["26n", "26s", "26m", "26l", "26x"]; sh = np.array([0.193, 0.205, 0.247, 0.23, 0.124]); sh /= sh.sum()
c = rng.choice(5, size=len(IMG_IDS), p=sh); L = np.array([MODELS[n][1] for n in names])
score(mix([names[i] for i in c]), f"  five-model random null at oracle shares, avgL={L[c].mean():.3f}")
