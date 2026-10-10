import numpy as np
H=[(13.93,.4937),(17.33,.5117),(21.92,.5262),(25.15,.5329),(28.31,.5368),(34.43,.5417)]
def E(g):
    for (a,pa),(b,pb) in zip(H,H[1:]):
        if a<=g<=b: return pa+(pb-pa)*(g-a)/(b-a)
rows=[("L512 crop",15.89,.5236,None),("L544 crop",18.01,.5303,.5231),("L576 crop",20.17,.5337,None),("L640 crop",24.63,.5380,.5305)]
for r in (2.25,3.1,4.8,0.8):
    print(f"router GMAC-eq {r}")
    for n,c,ap,nul in rows:
        g=c+r; print(f"  {n}: cost {g:.2f}  env {E(g):.4f}  route-env {ap-E(g):+.4f}  ms@0.15 {g/0.15+4:.0f}" + (f"  route-null {ap-nul:+.4f}" if nul else ""))
for n,g,ap in (("L448",17.33,.5117),("L512",21.92,.5262),("L544",25.15,.5329),("L576",28.31,.5368),("L640",34.43,.5417),("M640",27.67,.5261)):
    print(f"dense {n}: {g} GMAC  ms@0.15 {g/0.15+2:.0f}  AP {ap}")
