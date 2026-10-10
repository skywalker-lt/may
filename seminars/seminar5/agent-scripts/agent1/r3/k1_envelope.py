# K1 on the MEASURED dense envelope over (model, input scale). T4 unset basis (the packet's F; agent 7's latency basis):
# measured M640 5.36, M512 3.78, M768 6.97, L640 6.89; every other off-640 point est. by pixel scaling t640*(af+(1-af)p),
# af fitted on M512/M640 (est., no T4 pod). Routed rows pay the router on EVERY image (0.18 ms n320; 0.064 ms n192, whose AP
# was measured within 0.0002 of n320 in round 2). No If compile discount. Router: n320 ridge fitted on 24k train2017 images.
import sys, numpy as np; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1/r3"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1")
import lib3, mixlib as M, feats as F
af = (3.78 / 5.36 - 0.64) / (1 - 0.64)
T = {"M640": 5.36, "M512": 3.78, "M768": 6.97, "L640": 6.89}
AP = {}
for k in lib3.NAMES:
    if k == "RS2FED": continue
    m, s = k[0], int(k[1:])
    T.setdefault(k, (5.36 if m == "M" else 6.89) * (af + (1 - af) * (s / 640) ** 2))
EXTRA = [("N640", 1.65, 0.4060), ("S640", 2.75, 0.4795), ("S768", 3.62, 0.4834), ("X640", 12.41, 0.5691)]
src = {k: lib3.evaluated(n) for k, n in lib3.NAMES.items() if k != "RS2FED"}
ids, ns, nm, nl, _ = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
for k, e in src.items(): AP[k] = M.score([e], np.zeros(I, int))[0]
pts = sorted([(T[k], AP[k], k) for k in AP] + [(t, a, k) for k, t, a in EXTRA])
def env(t):
    best, arg = -1, None
    for i, (ta, a, ka) in enumerate(pts):
        for tb, b, kb in pts[i:]:
            if ta <= t <= tb:
                v = a if tb == ta else a + (b - a) * (t - ta) / (tb - ta)
                if v > best: best, arg = v, (ka, kb, (t - ta) / (tb - ta) if tb > ta else 0.0)
    return best, arg
Fr = lambda t: 0.5261 + 0.0102 * (t - 5.36)
# train-fitted routers
d = np.load("/data/tmp/ds-yolo/seminar5/work/agent1/r2/train_n320.npz", allow_pickle=True)
Xt = d["n320"].astype(np.float64); At = d["areas"]; k17 = 1.7
mu, sd = Xt.mean(0), Xt.std(0) + 1e-6; Z = (Xt - mu) / sd; Zv = (Xn - mu) / sd
def ridge(y, lam=10):
    w = np.linalg.solve(Z.T @ Z + lam * len(Z) / 100 * np.eye(Z.shape[1]), Z.T @ (y - y.mean())); return Zv @ w + y.mean()
shrink = ridge(np.array([np.log1p((a < k17 * 96**2).sum()) for a in At]))
count = ridge(np.array([np.log1p(len(a)) for a in At]))
gt_sm, gt_c = np.log1p(ns + nm), np.log1p(ns + nm + nl)
rng = np.random.RandomState(7)
def route(q, qh, lo_s, hi_s):
    """0 = lo rung (lowest shrink score, share q), 2 = hi rung (highest count among the rest, share qh), 1 = mid."""
    ch = np.ones(I, int)
    if qh > 0:
        o = np.lexsort((rng.rand(I), -hi_s)); ch[o[:int(round(qh * I))]] = 2
    rest = np.where(ch == 1)[0]; o2 = rest[np.lexsort((rng.rand(len(rest)), lo_s[rest]))]; ch[o2[:int(round(q * I))]] = 0
    return ch
def null(menu, q, qh, n=2):
    out = []
    for r in range(n):
        g = np.random.RandomState(100 + r); ch = np.ones(I, int); p = g.permutation(I)
        ch[p[:int(round(q * I))]] = 0
        if qh > 0: ch[p[int(round(q * I)):int(round((q + qh) * I))]] = 2
        out.append(M.score([src[m] for m in menu], ch)[0])
    return float(np.mean(out))
print("af", round(af, 3)); print("dense points (T4 unset ms, est. where not measured):")
for t, a, k in pts: print(f"  {k:5s} {t:5.2f} ms AP {a:.4f}  F+0.003 {a - Fr(t) - 0.003:+.4f}  on-envelope {abs(env(t)[0] - a) < 1e-9}")
rows = []
def row(menu, q, qh, rc, tag="learned", lo_s=None, hi_s=None):
    lo_s = shrink if lo_s is None else lo_s; hi_s = count if hi_s is None else hi_s
    if len(menu) == 2: ch = route(q, 0, lo_s, hi_s); ch[ch == 1] = 1; srcs = [src[menu[0]], src[menu[1]]]; sh = [q, 1 - q]; qh_ = 0
    else: ch = route(q, qh, lo_s, hi_s); srcs = [src[m] for m in menu]; sh = [q, 1 - q - qh, qh]; qh_ = qh
    a = M.score(srcs, ch)[0]; t = rc + sum(s * T[m] for s, m in zip(sh, menu)); wc = rc + max(T[m] for m in menu)
    e, arg = env(t); return a, t, wc, e, arg
def report(menu, q, qh=0.0, nulls=True):
    out = []
    for rc in (0.18, 0.064):
        a, t, wc, e, arg = row(menu, q, qh, rc)
        out.append((rc, a, t, wc, e, arg))
    n = null(menu if len(menu) == 2 else menu, q, qh) if nulls else float("nan")
    s = "/".join(menu)
    for rc, a, t, wc, e, arg in out:
        line = (f"{s:16s} q {q:.2f} qh {qh:.2f} rc {rc:.3f} | AP {a:.4f} avg {t:.2f} worst {wc:.2f} ({wc/5.36:.2f}x M) | (a) bar {a - Fr(t) - 0.003:+.4f}"
                f" | (b) null {n:.4f} -> {a - n - 0.003:+.4f} | (c) env {e:.4f} [{arg[0]}-{arg[1]} w {arg[2]:.2f}] -> {a - e - 0.003:+.4f}")
        print(line, flush=True); rows.append((s, q, qh, rc, a, t, wc, n, e, arg))
print("\n# two-rung menus, lo rung by predicted log1p(S+M) (shrink)")
for menu in (("M512", "M640"), ("M448", "M640"), ("M576", "M640"), ("L512", "L640"), ("L448", "L640"), ("L576", "L640"), ("L448", "L576"), ("L512", "L576"), ("M512", "L640"), ("M448", "L576"), ("M512", "L576"), ("M448", "L512"), ("L448", "L512")):
    for q in (0.3, 0.4, 0.5, 0.6, 0.7):
        report(menu, q)
print("\n# three-rung menus, hi rung by predicted log1p(count), lo by shrink among the rest")
for menu in (("M512", "M640", "L640"), ("M448", "L512", "L640"), ("L448", "L512", "L640"), ("L448", "L576", "L640"), ("M512", "L576", "L640"), ("M448", "L512", "L576"), ("L448", "L512", "L576")):
    for q, qh in ((0.3, 0.2), (0.4, 0.2), (0.5, 0.2), (0.5, 0.3), (0.6, 0.2), (0.6, 0.3), (0.4, 0.4)):
        report(menu, q, qh)
import pickle; pickle.dump(rows, open("/data/tmp/ds-yolo/seminar5/work/agent1/r3/k1_rows.pkl", "wb"))
