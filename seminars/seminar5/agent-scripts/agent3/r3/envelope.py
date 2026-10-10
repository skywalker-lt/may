# Dense envelope over (model, input scale), T4 unset basis. Latencies of the new points are est. by pixel scaling:
# t_M(s) = a + b (s/640)^2 fitted on M@512 3.78 / M@640 5.36 (measured unset); L scaled by M's relative curve
# from L@640 6.89 (measured). AP measured (one-to-one, protocol). Packet front F(t) = 0.5261 + 0.0102 (t - 5.36).
import numpy as np
b = (5.36 - 3.78) / (1 - 0.64); a = 5.36 - b
tM = lambda s: a + b * (s / 640) ** 2
tL = lambda s: 6.89 * tM(s) / 5.36
P = [('M448', tM(448), 0.4937), ('M512', 3.78, 0.5063), ('M576', tM(576), 0.5203), ('M608', tM(608), 0.5234),
     ('M640', 5.36, 0.5261), ('M768', 6.96, 0.5196), ('L448', tL(448), 0.5117), ('L512', tL(512), 0.5262),
     ('L576', tL(576), 0.5368), ('L640', 6.89, 0.5417), ('S640', 2.75, 0.4795), ('X640', 12.41, 0.5691)]
P.sort(key=lambda p: p[1])
hull = []
for p in P:
    while len(hull) >= 2:
        (n1, x1, y1), (n2, x2, y2) = hull[-2], hull[-1]
        if (y2 - y1) * (p[1] - x1) <= (p[2] - y1) * (x2 - x1): hull.pop()
        else: break
    hull.append(p)
# keep only the upper-left frontier (monotone increasing AP)
H = [hull[0]]
for p in hull[1:]:
    if p[2] > H[-1][2]: H.append(p)
F = lambda t: 0.5261 + 0.0102 * (t - 5.36)
def env(t):
    xs = [h[1] for h in H]; ys = [h[2] for h in H]
    return float(np.interp(t, xs, ys))
print('points (est. ms unset, AP, AP - F):')
for n, t, y in P: print(f'  {n:5s} {t:6.2f} {y:.4f} {y - F(t):+.4f}')
print('upper envelope:', ' -> '.join(f'{n}({t:.2f})' for n, t, _ in H))
for i in range(len(H) - 1): print(f'  slope {H[i][0]}->{H[i+1][0]}: {(H[i+1][2]-H[i][2])/(H[i+1][1]-H[i][1]):.4f} AP/ms')
print('envelope minus F at t:', ', '.join(f'{t}: {env(t)-F(t):+.4f}' for t in [4.0, 4.5, 4.87, 5.13, 5.36, 5.82, 6.0]))
rows = [  # (design, AP, avg ms est., source)
 ('1-bit B16 share 0.49 (tau16 proxy)', 0.5251, 5.36 - 0.49 * 1.00, 'r1 routes_noP3'),
 ('1-bit B32 share 0.49 (tau32 proxy)', 0.5187, 5.36 - 0.49 * 1.00, 'r1'),
 ('ladder B16/A/C768 0.49/0.20, C 1.3 ms', 0.5292, 5.36 - 0.49 + 0.20 * 1.3, 'r1 ladder3b'),
 ('ladder B16/A/Cleaf32 0.49/0.20 (agent 4 leaf proxy, est. AP)', 0.5276, 5.36 - 0.49 + 0.20 * 1.3, 'agent4 r2 cleaf'),
 ('B16 0.49 / A / L 0.30', 0.5318, 5.36 - 0.49 + 0.30 * 1.53, 'r2 cmp2'),
 ('B32 0.49 / A / L 0.30', 0.5253, 5.36 - 0.49 + 0.30 * 1.53, 'r2 cmp2'),
 ('exchange 512 0.5 / A / L 0.45 (router 0.18 on all)', 0.5308, 5.36 - 0.5 * 1.58 + 0.45 * 1.53 + 0.18, 'r2 cmp3'),
 ('exchange + 512-noP3(tau16) 0.2 (router on all)', 0.5305, 5.36 - 0.5 * 1.58 - 0.2 * 0.57 + 0.45 * 1.53 + 0.18, 'r2 cmp2'),
 ('RS-2 M 512/640 share 0.5 (router on all)', 0.5230, 5.36 - 0.5 * 1.58 + 0.18, 'agent1/rs2_fed'),
]
print('\ndesign | AP | est. ms | bar (F+0.003) margin | envelope+0.003 margin')
for d, y, t, s in rows:
    print(f'  {d:58s} {y:.4f} {t:5.2f} {y - F(t) - 0.003:+.4f} {y - env(t) - 0.003:+.4f}  [{s}]')
np.save('/data/tmp/ds-yolo/seminar5/work/agent3/r3/env_hull.npy', np.array([[h[1], h[2]] for h in H]))
