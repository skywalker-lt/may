"""Pred.: dense YOLO26-L (and M) at reduced input scale on the T4, from M's three measured AP points (512/640/768,
quadratic in ln s) and a latency model a + b (s/640)^2 fitted to M's measured 512/640 T4 times (unset basis 3.78/5.36);
L assumed to share M's fixed term a and M's relative AP-vs-scale curve. Compared with the packet's linear front."""
import numpy as np
x=np.log(np.array([512,640,768])/640); ap=np.array([0.5063,0.5261,0.5196]); c=np.polyfit(x,ap,2)
drop=lambda s: np.polyval(c,np.log(s/640))-0.5261
b=(5.36-3.78)/(1-0.64); a=5.36-b
F=lambda t: 0.5261+0.0102*(t-5.36)
for name,ap640,t640 in [('M',0.5261,5.36),('L',0.5417,6.89)]:
    bb=t640-a
    for s in [480,512,544,576,608,640]:
        t=a+bb*(s/640)**2; p=ap640+drop(s)
        print(f'{name}@{s}: T4 est. {t:.2f} ms  AP pred. {p:.4f}  front {F(t):.4f}  margin {p-F(t):+.4f}')
