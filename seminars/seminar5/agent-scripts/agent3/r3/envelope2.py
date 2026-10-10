# Continuous dense envelope: AP(s) per model by PCHIP in ln s through the measured scales; latency est. by pixel
# scaling (M: a + b (s/640)^2 on 3.78/5.36 unset; L: M's relative curve times 6.89). Upper concave hull over both
# curves (time-sharing between any two dense points is a dense null). T4 unset basis, all latencies est.
import numpy as np
from scipy.interpolate import PchipInterpolator as P
b = (5.36 - 3.78) / 0.36; a = 5.36 - b; tM = lambda s: a + b * (s / 640) ** 2; tL = lambda s: 6.89 * tM(s) / 5.36
Ms = [448, 512, 576, 608, 640, 768]; Ma = [0.4937, 0.5063, 0.5203, 0.5234, 0.5261, 0.5196]
Ls = [448, 512, 576, 640]; La = [0.5117, 0.5262, 0.5368, 0.5417]
fM = P(np.log(Ms), Ma); fL = P(np.log(Ls), La)
pts = [(tM(s), float(fM(np.log(s))), 'M', s) for s in np.arange(448, 769, 4)] + \
      [(tL(s), float(fL(np.log(s))), 'L', s) for s in np.arange(448, 641, 4)] + [(12.41, 0.5691, 'X', 640)]
pts.sort()
hull = []
for p in pts:
    while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]): hull.pop()
    hull.append(p)
H = [hull[0]]
for p in hull[1:]:
    if p[1] > H[-1][1]: H.append(p)
X = np.array([h[0] for h in H]); Y = np.array([h[1] for h in H])
env = lambda t: float(np.interp(t, X, Y))
np.save('env_cont.npy', np.stack([X, Y]))
F = lambda t: 0.5261 + 0.0102 * (t - 5.36)
if __name__ == '__main__':
    print('hull vertices:', ', '.join(f'{m}{s}({t:.2f},{y:.4f})' for t, y, m, s in H if t < 8))
    for t in [4.0, 4.2, 4.45, 4.6, 4.87, 5.13, 5.33, 5.52, 5.82]:
        print(f'  t {t:.2f}: env {env(t):.4f}  env-F {env(t) - F(t):+.4f}')
