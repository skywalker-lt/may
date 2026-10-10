# Upper concave hull of dense (model, scale) points, T4 unset basis; 640 points measured, others est. by agent 7's pixel scaling.
import numpy as np
P={'M448':(3.12,0.4937),'M512':(3.78,0.5063),'L448':(4.01,0.5117),'M576':(4.53,0.5203),'L512':(4.86,0.5262),
   'M608':(4.93,0.5234),'M640':(5.36,0.5261),'L576':(5.82,0.5368),'L640':(6.89,0.5417),'X640':(12.41,0.5691)}
pts=sorted(P.values())
h=[]
for p in pts:
    while len(h)>=2 and (h[-1][0]-h[-2][0])*(p[1]-h[-2][1])-(h[-1][1]-h[-2][1])*(p[0]-h[-2][0])>=0: h.pop()
    h.append(p)
names={v:k for k,v in P.items()}
print('hull vertices:',[names[p] for p in h])
def env(t): return np.interp(t,[p[0] for p in h],[p[1] for p in h])
F=lambda t:0.5261+0.0102*(t-5.36)
for t in [4.3,4.5,4.75,5.0,5.36,5.6,5.9,6.0,6.2,6.45]:
    print(f't={t:.2f} ms  F={F(t):.4f}  env={env(t):.4f}  env-F={env(t)-F(t):+.4f}  need(c)={env(t)+0.003:.4f}')
# TCR on M@640 (agent 5 T4 est. +0.52..0.86 ms; stub upper bound +0.67..1.10)
for d in [0.52,0.86,1.10]:
    t=5.36+d; print(f'TCR on M: t={t:.2f}  need over M for (a) {F(t)+0.003-0.5261:+.4f}  for (c) {env(t)+0.003-0.5261:+.4f}')
for d in [0.52,0.86,1.10]:
    t=4.86+d; print(f'TCR on L@512: t={t:.2f}  need over L512 for (a) {F(t)+0.003-0.5262:+.4f}  for (c) {env(t)+0.003-0.5262:+.4f}')
