# RS-2L prediction (pred.): public L at {512, 640}, assuming L's per-image scale structure equals M's (routed loss at each 512 share
# taken from M's measured learned route, t4_shrink.log); T4 real basis, L@512 est. = 6.55 x 3.47/5.05; vs linear front, vs the
# predicted dense envelope (M and L rescaled), and vs L's share null (M's null loss).
import numpy as np
c = np.polyfit(np.log(np.array([512, 640, 768]) / 640), [0.5063, 0.5261, 0.5196], 2); af = 0.661 / 5.05
def curve(t, ap, t0):
    p2 = (t / t0 - af) / (1 - af)
    return ap + np.polyval(c, 0.5 * np.log(p2)) - np.polyval(c, 0.0) if 0.3 < p2 <= 1.44 else -1
env = lambda t: max(curve(t, 0.5261, 5.05), curve(t, 0.5417, 6.55))
for s in (480, 512, 544, 576, 608):
    t = 6.55 * (af + (1 - af) * (s / 640) ** 2); print(f"dense L@{s}: est. {t:.2f} ms real, pred. {curve(t, 0.5417, 6.55):.4f}, linear front {0.5261 + 0.0102 * (t - 5.05):.4f}")
M_loss = {0.3: 0.5261 - 0.5250, 0.4: 0.5261 - 0.5242, 0.5: 0.5261 - 0.5230, 0.6: 0.5261 - 0.5216, 0.7: 0.5261 - 0.5192, 0.8: 0.5261 - 0.5154}
M_null = {0.3: 0.5261 - 0.5200, 0.4: 0.5261 - 0.5182, 0.5: 0.5261 - 0.5158, 0.6: 0.5261 - 0.5146, 0.7: 0.5261 - 0.5119, 0.8: 0.5261 - 0.5106}
tL512 = 6.55 * 3.47 / 5.05
for s, l in M_loss.items():
    t = s * tL512 + (1 - s) * 6.55; ap = 0.5417 - l; F = 0.5261 + 0.0102 * (t - 5.05)
    print(f"RS-2L share {s}: est. {t:.2f} ms real, pred. {ap:.4f} | vs linear bar {ap - F - 0.003:+.4f} | vs pred. envelope {ap - env(t):+.4f} | vs null {ap - (0.5417 - M_null[s]):+.4f}")
for s, l in M_loss.items():
    t = s * 3.47 + (1 - s) * 5.05; print(f"RS-2 (M) share {s}: {t:.2f} ms, measured {0.5261 - l:.4f} | vs pred. envelope {0.5261 - l - env(t):+.4f}")
