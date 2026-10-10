"""Content-rectangle route on a dynamic-shape CPU runtime (lens 3), from the dumps. CPU, one thread.
Per image: router = YOLO26-N@640 detections with score >= tau; rectangle = union of router boxes + 4% margin per side.
The big model (L or M) computes only inside the rectangle at native pixel density (filter approximation: keep its
dense detections whose centre lies in the rectangle). Optional mixture: N's detections outside the rectangle are kept.
Cost: per-image MACs with rect letterbox (long side S, each side ceil to 32), MAC(model, S) = G640 * W'H' / 640^2."""
import os, sys, json, time, math, numpy as np, contextlib, io
os.environ["OMP_NUM_THREADS"]="1"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D="/data/tmp/ds-yolo/seminar6/inputs/dumps/"; D2="/data/tmp/ds-yolo/seminar6/inputs/dumps_r2/"
coco=COCO(D+"instances_val2017.json")
imgs={i["id"]:(i["width"],i["height"]) for i in coco.loadImgs(coco.getImgIds())}
ids=sorted(imgs)
G640={'n':3.060,'s':11.419,'m':37.699,'l':46.899}
def load(p):
    by={}
    for r in json.load(open(p)): by.setdefault(r["image_id"],[]).append(r)
    return by
gt_by={}
for a in coco.loadAnns(coco.getAnnIds()): gt_by.setdefault(a["image_id"],[]).append(a["bbox"])
c32=lambda v: 32*math.ceil(v/32-1e-9)
def dense_mac(model,S,w,h):
    r=S/max(w,h); return G640[model]*c32(w*r)*c32(h*r)/640**2
def crop_mac(model,S,w,h,rect):
    if rect is None: return 0.0
    r=S/max(w,h); x0,y0,x1,y1=rect
    return G640[model]*c32((x1-x0)*r)*c32((y1-y0)*r)/640**2
def rect_of(boxes,w,h,margin=0.04):
    if not boxes: return None
    b=np.array(boxes); x0=b[:,0].min(); y0=b[:,1].min(); x1=(b[:,0]+b[:,2]).max(); y1=(b[:,1]+b[:,3]).max()
    mx=margin*w; my=margin*h
    return (max(0,x0-mx),max(0,y0-my),min(w,x1+mx),min(h,y1+my))
def centred(rect,w,h):
    if rect is None: return None
    rw=rect[2]-rect[0]; rh=rect[3]-rect[1]; x0=(w-rw)/2; y0=(h-rh)/2
    return (x0,y0,x0+rw,y0+rh)
inside=lambda d,R: R is not None and R[0]<=d["bbox"][0]+d["bbox"][2]/2<=R[2] and R[1]<=d["bbox"][1]+d["bbox"][3]/2<=R[3]
def evaluate(dets,label):
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        E=COCOeval(coco,coco.loadRes(dets),"bbox"); E.evaluate(); E.accumulate(); E.summarize()
    s=E.stats
    print(f"EVAL {label}: AP={s[0]:.4f} AP50={s[1]:.4f} AP75={s[2]:.4f} S/M/L={s[3]:.4f}/{s[4]:.4f}/{s[5]:.4f} n={len(dets)} ({time.time()-t:.0f}s)",flush=True)
    return dict(label=label,AP=s[0],AP50=s[1],AP75=s[2],APS=s[3],APM=s[4],APL=s[5])
def rects(router,tau):
    out={}
    for i in ids:
        w,h=imgs[i]
        rb=gt_by.get(i,[]) if router=="gt" else [r["bbox"] for r in router.get(i,[]) if r["score"]>=tau]
        out[i]=rect_of(rb,w,h)
    return out
def cost(model,S,R):
    d=np.array([dense_mac(model,S,*imgs[i]) for i in ids]); c=np.array([crop_mac(model,S,*imgs[i],R[i]) for i in ids])
    return d,c
def compose(big,R,outside=None):
    out=[]
    for i in ids:
        out+= [d for d in big.get(i,[]) if inside(d,R[i])]
        if outside is not None: out+= [d for d in outside.get(i,[]) if not inside(d,R[i])]
    return out
if __name__=="__main__":
    stage=sys.argv[1]
    N=load(D+"dump_yolo26n_coco.json")
    res=[]
    if stage=="cost":
        for tau in (0.05,0.1,0.25):
            R=rects(N,tau); d,c=cost('l',640,R)
            fr=c/d; empty=np.mean([R[i] is None for i in ids])
            print(f"tau {tau}: L rect-letterbox dense mean {d.mean():.2f} GMAC; crop mean {c.mean():.2f} (frac {c.sum()/d.sum():.3f}); "
                  f"per-image frac p50 {np.median(fr):.3f} p90 {np.percentile(fr,90):.3f} max {fr.max():.3f}; full-frame share {np.mean(fr>=0.999):.3f}; empty {empty:.3f}")
        R=rects("gt",0); d,c=cost('l',640,R); print(f"GT: crop frac {c.sum()/d.sum():.3f}")
        for m,S in (('n',640),('n',320),('m',448),('m',512),('m',576),('m',608),('m',640),('l',448),('l',512),('l',544),('l',576),('l',640)):
            d=np.array([dense_mac(m,S,*imgs[i]) for i in ids]); print(f"dense {m}@{S}: mean {d.mean():.2f} GMAC (square {G640[m]*(S/640)**2:.2f})")
        sys.exit()
    L=load(D+"dump_yolo26l_coco.json")
    R=rects(N,0.05)
    if stage=="L":
        res.append(evaluate(compose(L,R),"L crop tau0.05"))
        res.append(evaluate(compose(L,R,N),"L crop tau0.05 + N outside"))
        Rc={i:centred(R[i],*imgs[i]) for i in ids}
        res.append(evaluate(compose(L,Rc,N),"NULL L centred same-size + N outside"))
        res.append(evaluate(compose(L,rects("gt",0),N),"ORACLE L GT rect + N outside"))
    if stage=="more":
        L544=load(D2+"dumpml_yolo26l_544_coco.json"); M=load(D+"dumpml_yolo26m_coco.json")
        res.append(evaluate(compose(L544,R,N),"L544 crop tau0.05 + N outside"))
        res.append(evaluate(compose(M,R,N),"M crop tau0.05 + N outside"))
    json.dump(res,open(f"croproute_{stage}.json","w"),indent=1)
if __name__=="__main__" and sys.argv[1]=="l544":
    for S in (544,640):
        d,c=cost('l',S,R); print(f"L@{S}: dense {d.mean():.2f} crop {c.mean():.2f} frac {c.sum()/d.sum():.3f}; worst image crop {c.max():.2f}",flush=True)
    L544=load(D2+"dumpml_yolo26l_544_coco.json")
    Rc={i:centred(R[i],*imgs[i]) for i in ids}
    res.append(evaluate(compose(L544,Rc,N),"NULL L544 centred same-size + N outside"))
    res.append(evaluate(compose(L544,R),"L544 crop tau0.05 (no N outside)"))
    json.dump(res,open("croproute_l544.json","w"),indent=1)
