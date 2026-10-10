"""Margin of OCC-L@544 over the rect-basis envelope as a function of the phone cost model: t = t0 + G/theta per forward.
Route pays two forwards. theta_L = 0.2 GMAC/ms (est.), ratio = theta_L/theta_N."""
H=[(13.93,.4937),(17.33,.5117),(21.92,.5262),(25.15,.5329),(28.31,.5368),(34.43,.5417)]
def E(g):
    for (a,pa),(b,pb) in zip(H,H[1:]):
        if a<=g<=b: return pa+(pb-pa)*(g-a)/(b-a)
    return H[0][1] if g<H[0][0] else H[-1][1]
thL=0.2
rows=[("OCC-L@544 (N@640)",18.01,2.25,.5303),("OCC-L@544 (N@320, pred. AP -0.002)",18.01*0.97,0.58,.5283),("CAP 544/512 (N@640)",16.74,2.25,.5271)]
print("t0 ms | ratio | " + " | ".join(n for n,*_ in rows))
for t0 in (2,5,10,20):
    for ratio in (1.0,1.4,1.7,2.2):
        cells=[]
        for n,gc,gn,ap in rows:
            g=gc+gn*ratio+t0*thL   # extra forward's fixed term in L-eq GMAC
            ms=(t0+g/thL)
            cells.append(f"{ap-E(g):+.4f} @{ms:.0f}ms")
        print(f"{t0:3d} | {ratio:.1f} | "+" | ".join(cells))
print("dense L@512 ms:",2/1+21.92/thL+0, " L@544:",25.15/thL)
