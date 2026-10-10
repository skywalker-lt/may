"""Round-2 checks (one CPU thread): (1) random-position null for OCC-L@544; (2) budget cap variant (crop at 544 unless its
MACs exceed dense L@512's mean, then L@512 inside the same rectangle); (3) rect-letterbox CPU dense vs square H200 dump on
the 500 subset (agent 5's assumption); (4) GT centres outside the N rectangle; (5) N@320 rectangle coverage on 64 images."""
import os,sys,json; os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, contextlib, io
import croproute as C
from pycocotools.cocoeval import COCOeval
N=C.load(C.D+"dump_yolo26n_coco.json"); R=C.rects(N,0.05)
# (4) GT centres outside rect, by size
tot=0;out=0;outS=0;S=0
for i in C.ids:
    for (x,y,w,h) in C.gt_by.get(i,[]):
        tot+=1; small=w*h<32*32; S+=small
        r=R[i]
        if r is None or not (r[0]<=x+w/2<=r[2] and r[1]<=y+h/2<=r[3]): out+=1; outS+=small
print(f"GT centres outside N rect (tau0.05): {out}/{tot} = {out/tot:.4f}; small objects outside {outS}/{S} = {outS/S:.4f}",flush=True)
# (5) N@320 rectangles on 64 images vs N@640 rectangles
N320=C.load("dump_yolo26n_320_cpu_64.json"); sub64=sorted(N320)
cov640=cov320=n=0; a640=a320=0
for i in sub64:
    w,h=C.imgs[i]; r6=R[i]; r3=C.rect_of([r["bbox"] for r in N320[i] if r["score"]>=0.05],w,h)
    for (x,y,bw,bh) in C.gt_by.get(i,[]):
        n+=1; cx,cy=x+bw/2,y+bh/2
        cov640+= r6 is not None and r6[0]<=cx<=r6[2] and r6[1]<=cy<=r6[3]
        cov320+= r3 is not None and r3[0]<=cx<=r3[2] and r3[1]<=cy<=r3[3]
    a640+=C.crop_mac('l',544,w,h,r6)/C.dense_mac('l',544,w,h); a320+=C.crop_mac('l',544,w,h,r3)/C.dense_mac('l',544,w,h)
print(f"64 images: GT centre coverage N@640 {cov640/n:.3f} N@320 {cov320/n:.3f}; mean crop frac N@640 {a640/len(sub64):.3f} N@320 {a320/len(sub64):.3f}",flush=True)
# (3) subset: square H200 dump of L@544 on the 500 subset
L544=C.load(C.D2+"dumpml_yolo26l_544_coco.json"); d=json.load(open("realcrop_L544_500_1.json")); sub=d['sub']
with contextlib.redirect_stdout(io.StringIO()):
    E=COCOeval(C.coco,C.coco.loadRes([r for i in sub for r in L544.get(i,[])]),"bbox"); E.params.imgIds=sub; E.evaluate(); E.accumulate(); E.summarize()
print(f"SUBSET square H200 dump L@544: AP={E.stats[0]:.4f} AP50={E.stats[1]:.4f} AP75={E.stats[2]:.4f} S/M/L={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f} (CPU rect-letterbox dense was 0.5731)",flush=True)
# (1) random-position null
rng=np.random.default_rng(0); Rr={}
for i in C.ids:
    r=R[i]; w,h=C.imgs[i]
    if r is None: Rr[i]=None; continue
    rw=r[2]-r[0]; rh=r[3]-r[1]; x0=rng.uniform(0,w-rw); y0=rng.uniform(0,h-rh); Rr[i]=(x0,y0,x0+rw,y0+rh)
res=[C.evaluate(C.compose(L544,Rr,N),"NULL L544 random-position same-size + N outside")]
# (2) budget cap: crop at 544 unless crop MACs > 21.92 (dense L@512 rect mean); then L@512 in the rectangle
L512=C.load(C.D2+"dumpml_yolo26l_512_coco.json"); out=[]; cost=[]
for i in C.ids:
    w,h=C.imgs[i]; c=C.crop_mac('l',544,w,h,R[i])
    if c>21.92: big=L512; c=C.crop_mac('l',512,w,h,R[i])
    else: big=L544
    cost.append(c); out+=[r for r in big.get(i,[]) if C.inside(r,R[i])]+[r for r in N.get(i,[]) if not C.inside(r,R[i])]
cost=np.array(cost); print(f"CAP variant: mean crop GMAC {cost.mean():.2f}, max {cost.max():.2f}, share re-routed {np.mean(cost>21.92*0+1e9):.3f}",flush=True)
print(f"CAP variant: images sent to 512: {sum(1 for i in C.ids if C.crop_mac('l',544,*C.imgs[i],R[i])>21.92)/len(C.ids):.3f}",flush=True)
res.append(C.evaluate(out,"CAP L544/L512 crop + N outside"))
json.dump(res,open("r2_checks.json","w"),indent=1)
