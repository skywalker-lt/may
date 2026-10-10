"""Round 2: exchange menus over {M, L} x scale against three comparators at the same T4 average (unset basis, est. for unmeasured
scales): the linear front + 0.003, the DENSE envelope of every measured single operating point (M@s, L@s; upper concave hull,
interpolated) + 0.003, and the share null (3 draws). Monotone thumbnail count rule: sparsest -> cheapest member."""
import os, sys, re, glob, itertools, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from headroom import I, cnt_pred_all, cnt_pred_small
from mixlib import mix_ap, load
F = lambda L: 0.5261 + 0.0102 * (L - 5.36); R = 0.18; W = "/data/tmp/ds-yolo/seminar5/work/agent7/ev"
LB = float(os.environ.get("LBASIS", "1.0"))  # L-at-scale cost multiplier sensitivity (1.0 = est. ratio rule)
def Mcost(s): return {512: 3.78, 640: 5.36, 768: 6.97}.get(s, 3.78 + ((s / 640) ** 2 - 0.64) * 4.389)   # unset; measured at 512/640/768
def Lcost(s): return 6.89 if s == 640 else 6.89 * Mcost(s) / 5.36 * LB                                    # est. (only 640 measured)
pts = {}
for p in glob.glob(f"{W}/*.pkl"):
    nm = os.path.basename(p)[:-4]; m = re.search(r"yolo26([ml])(?:_(\d{3}))?_coco$", nm)
    if not m or "o2m" in nm or "nms" in nm: continue
    fam, s = m.group(1), int(m.group(2) or 640); key = f"{fam.upper()}@{s}"
    if key in pts and nm.startswith("dump_"): continue
    pts[key] = (nm, Mcost(s) if fam == "m" else Lcost(s))
for k in sorted(pts, key=lambda k: pts[k][1]): print(f"dense {k:7s} {pts[k][0]:32s} T4 {pts[k][1]:.2f} ms (unset{' est.' if k not in ('M@512','M@640','M@768','L@640') else ''}) AP {load(pts[k][0])['stats'][0]:.4f} vs F+0.003 {load(pts[k][0])['stats'][0]-F(pts[k][1])-0.003:+.4f}", flush=True)
hull = sorted((c, load(n)["stats"][0]) for n, c in pts.values())
def env(L):  # upper envelope of dense points, linear interpolation between neighbours (no routing implied: a dense model must pick one)
    best = max(a for c, a in hull if c <= L + 1e-9) if any(c <= L for c, a in hull) else -1
    for (c0, a0), (c1, a1) in itertools.combinations(hull, 2):
        if c0 <= L <= c1 and c1 > c0: best = max(best, a0 + (a1 - a0) * (L - c0) / (c1 - c0))
    return best
def best_dense_at_or_under(L): return max((a, k) for k, (n, c) in pts.items() for a in [load(n)["stats"][0]] if c <= L + 1e-9)
score = cnt_pred_all + 1e-6 * cnt_pred_small; rank = np.empty(I, int); rank[np.argsort(score + 1e-9 * np.arange(I))] = np.arange(I)
def assign(shares):
    cut = np.cumsum(shares)[:-1] * I; return np.searchsorted(cut, rank, side="right")
def show(keys, shares, tag):
    names = [pts[k][0] for k in keys]; c = np.array([pts[k][1] for k in keys]); a = assign(shares); L = c[a].mean() + R
    ap = mix_ap(names, a); nl = np.mean([mix_ap(names, np.random.RandomState(400 + d).permutation(a)) for d in range(2)]); e = env(L); bd = best_dense_at_or_under(L)
    print(f"{tag:34s} {'/'.join(keys):26s} shares {np.round(np.bincount(a, minlength=len(keys))/I,2)} avg {L:.2f} AP {ap:.4f} | -F-0.003 {ap-F(L)-0.003:+.4f} | -env-0.003 {ap-e-0.003:+.4f} (env {e:.4f}; best single <= avg {bd[1]} {bd[0]:.4f}) | -null {ap-nl:+.4f}", flush=True)
def budget2(k0, k1, target=5.36):
    c0, c1 = pts[k0][1], pts[k1][1]; q = (c1 + R - target) / (c1 - c0); return [q, 1 - q] if 0 < q < 1 else None
if __name__ == "__main__":
    T = float(os.environ.get("TARGET", "5.36")); have = sorted(pts, key=lambda k: pts[k][1])
    for k0, k1 in itertools.combinations(have, 2):
        sh = budget2(k0, k1, T)
        if sh: show([k0, k1], sh, f"two members at {T}")
    for trip in (("M@512","M@640","L@640"),("M@448","M@640","L@640"),("M@512","L@576","L@640"),("M@448","L@576","L@640"),("L@448","L@576","L@640"),("L@512","L@576","L@640"),("L@448","L@512","L@640"),("M@448","L@512","L@640")):
        c = [pts[k][1] for k in trip]
        for q in (0.2, 0.33, 0.5):
            qe = (q * (c[1] - c[0]) - R - (c[1] - T)) / (c[2] - c[1])
            if 0 < qe < 1 - q: show(list(trip), [q, 1 - q - qe, qe], f"three members q={q} at {T}")
