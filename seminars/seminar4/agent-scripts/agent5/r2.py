#!/usr/bin/env python3
"""Agent 5, round 2: (1) proxy cross-check (my 101-pt proxy vs agent 1's TP-0.5FP proxy) on the same dumps;
(2) equal-cost noise null {26m, v12m, 11m}; (3) resolution oracle with the new 512/768 dumps;
(4) shared-stem m/l at the measured 4.85/6.06 ms branch costs. All mixtures scored with full pycocotools."""
import sys, json, os
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, "/data/tmp/ds-yolo/seminar4/work/agent5")
import oracle as O
from oracle import MODELS, W, D, IMG_IDS, gt, mix, score

MODELS.update({"m512": ("dumpml_yolo26m_512_coco.json", 3.781), "m768": ("dumpml_yolo26m_768_coco.json", 6.959),
               "s768": ("dumpml_yolo26s_768_coco.json", 3.617), "m640": ("dumpml_yolo26m_coco.json", 5.355)})


def proxy_a1(ev, thr=0.2):
    """Agent 1's proxy: mean over IoU thresholds of (TP - 0.5 FP) over detections with score >= thr."""
    T = len(ev.params.iouThrs); K = len(ev.params.catIds); I = len(ev.params.imgIds)
    out = np.zeros(I)
    for i in range(I):
        acc = np.zeros(T)
        for k in range(K):
            e = ev.evalImgs[k * I + i]
            if e is None or not len(e["dtScores"]):
                continue
            s = np.asarray(e["dtScores"]); keep = s >= thr
            m = np.asarray(e["dtMatches"])[:, keep]; ig = np.asarray(e["dtIgnore"], bool)[:, keep]
            acc += ((m > 0) & ~ig).sum(1) - 0.5 * ((m == 0) & ~ig).sum(1)
        out[i] = acc.mean()
    return out


def ensure(name, a1=False):
    f = f"{W}/pi_{name}.npy"; fa = f"{W}/pa1_{name}.npy"
    if os.path.exists(f) and (not a1 or os.path.exists(fa)):
        return
    ev = score(O.load(name), f"single {name}", full=False)
    if not os.path.exists(f):
        np.save(f, O.per_image_ap(ev))
    if a1:
        np.save(fa, proxy_a1(ev))


def lagr(P, L, B, names, tag):
    best = None
    for lam in np.concatenate([[0], np.logspace(-4, 0, 200)]):
        c = np.argmax(P - lam * L[None, :] - 1e-9 * L[None, :], 1)
        avgL = L[c].mean()
        if avgL <= B and (best is None or P[np.arange(len(c)), c].sum() > best[0]):
            best = (P[np.arange(len(c)), c].sum(), lam, c, avgL)
    _, lam, c, avgL = best
    sh = {n: round(float((c == i).mean()), 3) for i, n in enumerate(names)}
    score(mix([names[i] for i in c]), f"{tag}: budget {B} avgL={avgL:.3f} shares={sh}")
    return c


def null(names, c, tag, seed=0):
    rng = np.random.default_rng(seed); L = np.array([MODELS[n][1] for n in names])
    sh = np.array([(c == i).mean() for i in range(len(names))])
    r = rng.choice(len(names), size=len(c), p=sh / sh.sum())
    score(mix([names[i] for i in r]), f"{tag} random null at those shares avgL={L[r].mean():.3f}")


part = sys.argv[1] if len(sys.argv) > 1 else "all"
five = ["26n", "26s", "26m", "26l", "26x"]
if part in ("all", "proxy"):
    for n in five:
        ensure(n, a1=True)
    Pm = np.stack([np.load(f"{W}/pi_{n}.npy") for n in five], 1)
    Pa = np.stack([np.load(f"{W}/pa1_{n}.npy") for n in five], 1)
    L = np.array([MODELS[n][1] for n in five])
    print("== (1) same dumps, same scoring, two proxies, budget 5.36 (standalone costs)")
    lagr(Pm, L, 5.36, five, "my 101-pt proxy")
    lagr(Pa, L, 5.36, five, "agent-1 TP-0.5FP proxy")
    lagr(Pm, L, 99, five, "my proxy, no budget")
    lagr(Pa, L, 99, five, "agent-1 proxy, no budget")
if part in ("all", "null"):
    for n in ["11m", "12m"]:
        ensure(n)
    eq = ["26m", "12m", "11m"]
    P = np.stack([np.load(f"{W}/pi_{n}.npy") for n in eq], 1); L = np.array([MODELS[n][1] for n in eq])
    print("== (2) equal-cost noise null: best-of {26m, v12m, 11m} (none better than 26m)")
    lagr(P, L, 99, eq, "equal-cost oracle {26m,12m,11m}")
    lagr(P[:, :2], L[:2], 99, eq[:2], "equal-cost oracle {26m,12m}")
if part in ("all", "res"):
    for n in ["m512", "m768", "s768", "m640"]:
        ensure(n)
    print("== (3) resolution routing with the new T4 dumps (multi-label, 512/640/768 costs 3.781/5.355/6.959; s768 3.617)")
    res = ["m512", "m640", "m768"]
    P = np.stack([np.load(f"{W}/pi_{n}.npy") for n in res], 1); L = np.array([MODELS[n][1] for n in res])
    for B in (5.36, 4.5):
        c = lagr(P, L, B, res, "oracle m@512/640/768")
        null(res, c, "res")
    res4 = res + ["s768"]
    P4 = np.stack([np.load(f"{W}/pi_{n}.npy") for n in res4], 1); L4 = np.array([MODELS[n][1] for n in res4])
    c = lagr(P4, L4, 5.36, res4, "oracle m@512/640/768 + s@768")
    null(res4, c, "res4")
    # ground-truth rules for the scale router (upper bounds on a thumbnail router estimating count / size)
    cnt = np.array([len(gt.getAnnIds(imgIds=i, iscrowd=False)) for i in IMG_IDS])
    med = np.array([np.median([a["area"] for a in gt.loadAnns(gt.getAnnIds(imgIds=i, iscrowd=False))] or [0]) for i in IMG_IDS])
    g = P[:, 2] - P[:, 0]
    print(f"768-512 gain: spearman(count)={spearmanr(cnt, g).correlation:+.3f} spearman(median area)={spearmanr(med, g).correlation:+.3f}")
    # two-scale 512/768: 768 share p so that avg = 5.36
    p = (5.36 - 3.781) / (6.959 - 3.781); n768 = int(p * len(IMG_IDS))
    for key, order in (("oracle", np.argsort(-g)), ("most objects->768", np.argsort(-cnt, kind="stable")),
                       ("smallest median->768", np.argsort(med, kind="stable"))):
        sel = np.zeros(len(IMG_IDS), bool); sel[order[:n768]] = True
        score(mix(np.where(sel, "m768", "m512")), f"  512/768 two-scale, 768 share {p:.3f}, {key}")
    rng = np.random.default_rng(0); sel = rng.random(len(IMG_IDS)) < p
    score(mix(np.where(sel, "m768", "m512")), f"  512/768 random null, 768 share {sel.mean():.3f}")
if part in ("all", "ml"):
    print("== (4) shared-stem M/L at the measured branch costs 4.85 / 6.06 ms (if_depth_ml); l share 0.42 for 5.36")
    for n in ["26m", "26l"]:
        ensure(n)
    Pm, Pl = np.load(f"{W}/pi_26m.npy"), np.load(f"{W}/pi_26l.npy"); g = Pl - Pm
    cnt = np.array([len(gt.getAnnIds(imgIds=i, iscrowd=False)) for i in IMG_IDS])
    p = (5.36 - 4.85) / (6.06 - 4.85); nl = int(p * len(IMG_IDS))
    for key, order in (("oracle", np.argsort(-g)), ("most objects->l", np.argsort(-cnt, kind="stable"))):
        sel = np.zeros(len(IMG_IDS), bool); sel[order[:nl]] = True
        score(mix(np.where(sel, "26l", "26m")), f"  m/l shared stem, l share {p:.3f}, {key}")
    rng = np.random.default_rng(0); sel = rng.random(len(IMG_IDS)) < p
    score(mix(np.where(sel, "26l", "26m")), f"  m/l random null, l share {sel.mean():.3f}")
    # s/l tails at agent-1-style shared-stem costs: s tail 1.0 est -> 3.7 ms, l 6.06: l share for 5.36
    Ps = np.load(f"{W}/pi_26s.npy"); g2 = Pl - Ps
    p2 = (5.36 - 3.7) / (6.06 - 3.7); n2 = int(p2 * len(IMG_IDS))
    for key, order in (("oracle", np.argsort(-g2)), ("most objects->l", np.argsort(-cnt, kind="stable"))):
        sel = np.zeros(len(IMG_IDS), bool); sel[order[:n2]] = True
        score(mix(np.where(sel, "26l", "26s")), f"  s/l shared stem (s tail 3.7 est, l 6.06), l share {p2:.3f}, {key}")
