"""Dense (model, scale) envelope on a MAC basis (the edge cost model: time = MACs / throughput + small fixed term).
Conv MACs scale with pixels exactly; the P5 attention (C2PSA, 0.4-0.6 GMAC at 640) scales with tokens^2 and is
folded in with pixel scaling (error < 0.2 GMAC). GMAC at 640 from flops_per_layer.py (thop, this fork's weights)."""
import numpy as np, json
G640={'n':3.060,'s':11.419,'m':37.699,'l':46.899}
AP={('n',640):0.4060,('s',640):0.4795,('m',448):0.4937,('m',512):0.5063,('m',576):0.5203,('m',608):0.5234,('m',640):0.5261,
    ('l',448):0.5117,('l',512):0.5262,('l',544):0.5329,('l',576):0.5368,('l',640):0.5417}
pts=sorted([(G640[m]*(s/640)**2,ap,f"{m.upper()}@{s}") for (m,s),ap in AP.items()])
print("point            GMAC    AP")
for g,ap,n in pts: print(f"{n:8s} {g:8.2f} {ap:.4f}")
# upper envelope (monotone hull of best AP at <= cost)
def env(g):
    best=-1
    for g1,ap1,_ in pts:
        if g1<=g: best=max(best,ap1)
    # linear interpolation between the hull points bracketing g
    hull=[]
    for g1,ap1,n in pts:
        if not hull or ap1>hull[-1][1]: hull.append((g1,ap1,n))
    for (ga,aa,_),(gb,ab,_) in zip(hull,hull[1:]):
        if ga<=g<=gb: return aa+(ab-aa)*(g-ga)/(gb-ga)
    return best
hull=[]
for g1,ap1,n in pts:
    if not hull or ap1>hull[-1][1]: hull.append((g1,ap1,n))
print("hull:",[(n,round(g,1),ap) for g,ap,n in hull])
json.dump(dict(points=pts,hull=hull),open("envelope_mac.json","w"),indent=1)
if __name__=="__main__":
    for g in (8,10,12,15,18,20,23,26,29,32,35,38,42,47):
        print(f"envelope at {g:5.1f} GMAC: {env(g):.4f}  (bar +0.003: {env(g)+0.003:.4f})")
