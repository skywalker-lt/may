"""Dense envelope with unmeasured scales filled in: per family, AP piecewise-linear in ln(scale) between measured scales
(a concave curve lies on or above this, so this envelope is still a LOWER bound on the dense comparator); cost from agent 7's
pixel model (est.). Prints the envelope at given latencies and compares with agent 7's measured-point chord."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7/r2")
import menus as M7
from mixlib import load
fam = {"M": {}, "L": {}}
for k, (n, c) in M7.pts.items():
    f, s = k.split("@"); fam[f][int(s)] = load(n)["stats"][0]
def dense_curve(f, s):
    ss = sorted(fam[f]); x = np.log(ss); y = [fam[f][t] for t in ss]
    return float(np.interp(np.log(s), x, y)) if ss[0] <= s <= ss[-1] else None
def cost(f, s): return M7.Mcost(s) if f == "M" else (6.89 if s == 640 else 6.89 * M7.Mcost(s) / 5.36 * M7.LB)
def env2(L):
    best = (-1, None)
    for f in fam:
        for s in range(448, 769, 4):
            a = dense_curve(f, s)
            if a is not None and cost(f, s) <= L + 1e-9 and a > best[0]: best = (a, f"{f}@{s}")
    return best
if __name__ == "__main__":
    for L in (4.3, 4.54, 4.70, 4.86, 5.0, 5.2, 5.36, 5.6, 5.82, 6.0, 6.3):
        a, k = env2(L); print(f"T4 {L:.2f} ms (unset, est.): chord env {M7.env(L):.4f} | ln-s interpolated env {a:.4f} ({k}) | F {M7.F(L):.4f}")
