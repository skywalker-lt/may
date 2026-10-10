# Pred. envelope with interpolated scales (every 16 px) from the measured AP points, piecewise quadratic in ln(s) through
# the three nearest measured scales; latency by the same pixel-scaling est. (af from M512/M640). Compared with the measured-point envelope.
import numpy as np
af = (3.78 / 5.36 - 0.64) / (1 - 0.64)
Mpts = {448: 0.4937, 512: 0.5063, 576: 0.5203, 608: 0.5234, 640: 0.5261, 768: 0.5196}
Lpts = {448: 0.5117, 512: 0.5262, 576: 0.5368, 640: 0.5417}
def interp(pts, s):
    ks = sorted(pts, key=lambda k: abs(np.log(k / s)))[:3]; c = np.polyfit(np.log(np.array(ks) / 640), [pts[k] for k in ks], 2)
    return np.polyval(c, np.log(s / 640))
def lat(m, s):
    if (m, s) in {("M", 640): 0, ("M", 512): 0, ("M", 768): 0, ("L", 640): 0}: return {("M", 640): 5.36, ("M", 512): 3.78, ("M", 768): 6.97, ("L", 640): 6.89}[(m, s)]
    return (5.36 if m == "M" else 6.89) * (af + (1 - af) * (s / 640) ** 2)
extra = [(1.65, 0.4060), (2.75, 0.4795), (12.41, 0.5691)]
meas = [(lat("M", s), a) for s, a in Mpts.items()] + [(lat("L", s), a) for s, a in Lpts.items()] + extra
pred = list(meas)
for s in range(448, 641, 16):
    if s not in Mpts: pred.append((lat("M", s), interp(Mpts, s)))
    if s not in Lpts: pred.append((lat("L", s), interp(Lpts, s)))
def env(t, P):
    P = sorted(P); best = -1
    for i, (ta, a) in enumerate(P):
        for tb, b in P[i:]:
            if ta <= t <= tb: best = max(best, a if tb == ta else a + (b - a) * (t - ta) / (tb - ta))
    return best
for s in (464, 480, 496, 528, 544, 560): print(f"L@{s}: est. {lat('L', s):.2f} ms, pred. AP {interp(Lpts, s):.4f}")
for t in (3.9, 4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.75, 4.9, 5.1, 5.36):
    print(f"t {t:.2f}: measured-point envelope {env(t, meas):.4f} | interpolated-scale envelope (pred.) {env(t, pred):.4f} | diff {env(t, pred) - env(t, meas):+.4f}")
print("# with extrapolated scales below 448 (pred.; quadratic in ln s through 448/512/576)")
pred2 = list(pred)
for s in (384, 400, 416, 432):
    pred2.append((lat("M", s), interp(Mpts, s))); pred2.append((lat("L", s), interp(Lpts, s)))
    print(f"  M@{s}: est. {lat('M', s):.2f} ms pred. {interp(Mpts, s):.4f} | L@{s}: est. {lat('L', s):.2f} ms pred. {interp(Lpts, s):.4f}")
for t in (3.75, 3.9, 4.0, 4.1, 4.2, 4.3):
    print(f"t {t:.2f}: measured {env(t, meas):.4f} | interpolated {env(t, pred):.4f} | interpolated + extrapolated {env(t, pred2):.4f} (+{env(t, pred2) - env(t, meas):.4f})")
