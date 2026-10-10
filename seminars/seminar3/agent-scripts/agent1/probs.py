# Monte Carlo for P(bar) and P(delta>=0.003). low/high read as 10th/90th percentiles of a split normal.
import numpy as np, sys
rng = np.random.default_rng(0); N = 400000; z = 1.2816
def sn(lo, c, hi):
    u = rng.standard_normal(N)
    return np.where(u < 0, c + u * (c - lo) / z, c + u * (hi - c) / z)
def run(name, ctrl, delta, bars):
    a = sn(*ctrl) + sn(*delta)
    q = np.percentile(a, [10, 50, 90])
    s = " ".join(f"P(>={b})={np.mean(a >= b):.3f}" for b in bars)
    print(f"{name:34s} AP p10/p50/p90 {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}  {s}  P(d>=.003)={np.mean(sn(*delta) >= 0.003):.3f}")
C80 = (0.516, 0.522, 0.527); C600 = (0.506, 0.515, 0.523)
print("-- agent 4 check: control 0.516/0.522/0.527, A delta -0.003/+0.001/+0.004")
run("agent4 A 80ep", C80, (-0.003, 0.001, 0.004), [0.5273, 0.5288])
print("-- agent 2 check: control 0.518/0.523/0.527 (central), A abs given; implied delta 0")
run("agent2 A 80ep (delta -0.003/0/+.003)", (0.518, 0.523, 0.527), (-0.003, 0.0, 0.003), [0.5273, 0.5288])
print("-- agent 1 round 2")
for nm, d80, d600, bars in [
    ("A-tied hard top-1 ST", (-0.006, -0.001, 0.003), (-0.008, 0.0, 0.006), [0.5273, 0.5288]),
    ("A6 equal-weight pairs", (-0.004, 0.0, 0.004), (-0.006, 0.001, 0.007), [0.5273, 0.5288]),
    ("B 8 outputs", (-0.002, 0.0, 0.002), (-0.003, 0.001, 0.004), [0.5328]),
    ("C scale+shift 39", (-0.002, 0.0005, 0.003), (-0.003, 0.001, 0.005), [0.5361]),
    ("D top-2 MatMul", (-0.004, 0.0005, 0.005), (-0.006, 0.0015, 0.008), [0.5370]),
    ("E dense", (0, 0, 0), (0, 0, 0), [0.5292]),
]:
    run(nm + " 80ep", C80, d80, bars); run(nm + " 600ep", C600, d600, bars)
