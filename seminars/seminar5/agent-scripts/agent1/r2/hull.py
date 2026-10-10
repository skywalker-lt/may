# The dense null with the scale knob, done properly: the upper concave hull of dense (model, scale) points on the T4 real basis
# (a random router between two dense points realises their chord, so the hull is the generalised share null). M and L off-640
# points are pred. from M's measured scale curve; M@512/640/768, L@640, S@640/768 are measured AP. Real-basis latency: unset x 0.942.
import numpy as np
c = np.polyfit(np.log(np.array([512, 640, 768]) / 640), [0.5063, 0.5261, 0.5196], 2); af = 0.661 / 5.05
d = lambda s: np.polyval(c, np.log(s / 640)) - np.polyval(c, 0.0)
pts = [(2.75 * 0.942, 0.4795), (3.62 * 0.942, 0.4834), (3.47, 0.5063), (5.05, 0.5261), (6.49, 0.5196), (6.55, 0.5417), (12.41 * 0.942, 0.5691)]
for s in range(448, 769, 32):
    if s not in (512, 640, 768): pts.append((5.05 * (af + (1 - af) * (s / 640) ** 2), 0.5261 + d(s)))
for s in range(448, 641, 32):
    if s != 640: pts.append((6.55 * (af + (1 - af) * (s / 640) ** 2), 0.5417 + d(s)))
pts = sorted(pts)
def hull(t, P=pts):
    best = -1
    for i, (ta, a) in enumerate(P):
        for tb, b in P[i:]:
            if ta <= t <= tb: best = max(best, a if tb == ta else a + (b - a) * (t - ta) / (tb - ta))
    return best
def hull_measured(t):  # hull over measured points only (no predicted off-640 L / M points)
    return hull(t, sorted([p for p in pts if p in [(2.75 * 0.942, 0.4795), (3.62 * 0.942, 0.4834), (3.47, 0.5063), (5.05, 0.5261), (6.49, 0.5196), (6.55, 0.5417), (12.41 * 0.942, 0.5691)]]))
rows = [("RS-2 M share 0.3 (meas.)", 4.58, 0.5250), ("RS-2 M share 0.5 (meas.)", 4.26, 0.5230), ("RS-2 M share 0.6 (meas.)", 4.10, 0.5216), ("RS-2 M share 0.7 (meas.)", 3.94, 0.5192),
        ("exchange q.4 qL.2 count (meas.)", 4.72, 0.5285), ("exchange q.4 qL.2 L random (meas.)", 4.72, 0.5274), ("exchange q.5 qL.3 count (meas.)", 4.71, 0.5296), ("exchange q.5 qL.3 L random (meas.)", 4.71, 0.5288),
        ("agent 7 exchange at dense-M latency (meas.)", 5.05, 0.5300),
        ("RS-2L share 0.5 (pred.)", 5.53, 0.5386), ("RS-2L share 0.6 (pred.)", 5.32, 0.5372), ("RS-2L share 0.7 (pred.)", 5.12, 0.5348), ("RS-2L share 0.8 (pred.)", 4.91, 0.5310)]
for n, t, a in rows:
    print(f"{n:45s} {t:.2f} ms AP {a:.4f} | linear F+0.003 {a - 0.5261 - 0.0102 * (t - 5.05) - 0.003:+.4f} | pred. dense hull {hull(t):.4f} -> {a - hull(t):+.4f} | measured-points hull {hull_measured(t):.4f} -> {a - hull_measured(t):+.4f}")
print("hull at", {t: round(hull(t), 4) for t in (4.0, 4.26, 4.5, 4.71, 5.05, 5.32, 5.53)})
