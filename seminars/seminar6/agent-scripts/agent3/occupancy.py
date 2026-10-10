"""Occupancy-sparse headroom from the dumps (CPU, one thread).
A tile grid GxG over each val2017 image; a tile is 'occupied' if a router box (N@640 detections at score>=tau,
or ground truth) intersects it, optionally dilated by one tile (halo). The dense model's detections whose centre
falls in a non-occupied tile are removed (the sparse trunk never computes there). AP via pycocotools."""
import os, sys, json, time, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import contextlib, io
D="/data/tmp/ds-yolo/seminar6/inputs/dumps/"
GT=D+"instances_val2017.json"
coco=COCO(GT)
imgs={i["id"]:(i["width"],i["height"]) for i in coco.loadImgs(coco.getImgIds())}
ids=sorted(imgs)
def load(p):
    d=json.load(open(p)); by={}
    for r in d: by.setdefault(r["image_id"],[]).append(r)
    return by
gt_by={}
for a in coco.loadAnns(coco.getAnnIds()):
    gt_by.setdefault(a["image_id"],[]).append(a["bbox"])

def tiles_hit(boxes,w,h,G):
    m=np.zeros((G,G),bool)
    for (x,y,bw,bh) in boxes:
        x0=int(np.clip(np.floor(x/w*G),0,G-1)); x1=int(np.clip(np.floor((x+bw)/w*G),0,G-1))
        y0=int(np.clip(np.floor(y/h*G),0,G-1)); y1=int(np.clip(np.floor((y+bh)/h*G),0,G-1))
        m[y0:y1+1,x0:x1+1]=True
    return m
def dilate(m):
    out=m.copy(); G=m.shape[0]
    for dy in (-1,0,1):
        for dx in (-1,0,1):
            out|=np.roll(np.roll(m,dy,0),dx,1) if True else m
    # roll wraps around; fix wrap by recomputing without wrap
    out=np.zeros_like(m)
    for i in range(G):
        for j in range(G):
            if m[max(0,i-1):i+2,max(0,j-1):j+2].any(): out[i,j]=True
    return out
def rect_mask(boxes,w,h):
    if not boxes: return None
    b=np.array(boxes); x0=b[:,0].min(); y0=b[:,1].min(); x1=(b[:,0]+b[:,2]).max(); y1=(b[:,1]+b[:,3]).max()
    return (max(0,x0),max(0,y0),min(w,x1),min(h,y1))

def evaluate(dets,label):
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        dt=coco.loadRes(dets) if dets else None
        E=COCOeval(coco,dt,"bbox"); E.evaluate(); E.accumulate(); E.summarize()
    s=E.stats
    print(f"EVAL {label}: AP={s[0]:.4f} AP50={s[1]:.4f} AP75={s[2]:.4f} S/M/L={s[3]:.4f}/{s[4]:.4f}/{s[5]:.4f} n={len(dets)} ({time.time()-t:.0f}s)",flush=True)
    return dict(AP=s[0],AP50=s[1],AP75=s[2],APS=s[3],APM=s[4],APL=s[5],n=len(dets))

def run(dense,router,G,tau,halo,mode,label,do_eval=True,topk=None):
    kept=[];frac=[];out=[]
    for i in ids:
        w,h=imgs[i]
        if router=="gt": rb=gt_by.get(i,[])
        else: rb=[r["bbox"] for r in router.get(i,[]) if r["score"]>=tau]
        dd=dense.get(i,[])
        if mode=="rect":
            r=rect_mask(rb,w,h)
            if r is None: frac.append(0.0); continue
            x0,y0,x1,y1=r
            # expand by margin of 4% of the image side (context), clip
            mx=0.04*w; my=0.04*h; x0=max(0,x0-mx); y0=max(0,y0-my); x1=min(w,x1+mx); y1=min(h,y1+my)
            frac.append((x1-x0)*(y1-y0)/(w*h))
            for d in dd:
                cx=d["bbox"][0]+d["bbox"][2]/2; cy=d["bbox"][1]+d["bbox"][3]/2
                if x0<=cx<=x1 and y0<=cy<=y1: out.append(d)
        else:
            if topk is not None:
                # occupancy score per tile = sum of router scores of boxes hitting the tile; keep top-k tiles
                sc=np.zeros((G,G))
                for r in router.get(i,[]):
                    if r["score"]<tau: continue
                    mm=tiles_hit([r["bbox"]],w,h,G); sc+=mm*r["score"]
                flat=sc.ravel(); order=np.argsort(-flat); m=np.zeros(G*G,bool); m[order[:topk]]=True; m=m.reshape(G,G)
            else:
                m=tiles_hit(rb,w,h,G)
                if halo: m=dilate(m)
            frac.append(m.mean())
            for d in dd:
                cx=d["bbox"][0]+d["bbox"][2]/2; cy=d["bbox"][1]+d["bbox"][3]/2
                tx=int(np.clip(cx/w*G,0,G-1)); ty=int(np.clip(cy/h*G,0,G-1))
                if m[ty,tx]: out.append(d)
    frac=np.array(frac)
    print(f"MASK {label}: keep mean={frac.mean():.3f} median={np.median(frac):.3f} p90={np.percentile(frac,90):.3f} max={frac.max():.3f} share_full={np.mean(frac>=0.999):.3f}",flush=True)
    res=dict(label=label,keep_mean=float(frac.mean()),keep_p90=float(np.percentile(frac,90)),keep_max=float(frac.max()))
    if do_eval: res.update(evaluate(out,label))
    return res

if __name__=="__main__":
    M=load(D+"dumpml_yolo26m_coco.json"); N=load(D+"dump_yolo26n_coco.json"); L=load(D+"dump_yolo26l_coco.json")
    print("loaded",flush=True)
    results=[]
    stage=sys.argv[1] if len(sys.argv)>1 else "frac"
    if stage=="frac":
        for G in (4,8):
            for tau in (0.05,0.1,0.25):
                for halo in (0,1):
                    results.append(run(M,N,G,tau,halo,"tile",f"M G{G} tau{tau} halo{halo}",do_eval=False))
            results.append(run(M,"gt",G,0,0,"tile",f"M G{G} GT halo0",do_eval=False))
            results.append(run(M,"gt",G,0,1,"tile",f"M G{G} GT halo1",do_eval=False))
        for tau in (0.05,0.1,0.25): results.append(run(M,N,0,tau,0,"rect",f"M rect tau{tau}",do_eval=False))
        results.append(run(M,"gt",0,0,0,"rect","M rect GT",do_eval=False))
    elif stage=="eval":
        results.append(evaluate([d for i in ids for d in M.get(i,[])],"M dense"))
        results.append(run(M,N,4,0.05,1,"tile","M G4 tau0.05 halo1"))
        results.append(run(M,N,4,0.1,0,"tile","M G4 tau0.1 halo0"))
        results.append(run(M,N,8,0.05,1,"tile","M G8 tau0.05 halo1"))
        results.append(run(M,"gt",4,0,1,"tile","M G4 GT halo1"))
        results.append(run(M,N,0,0.05,0,"rect","M rect tau0.05"))
        results.append(run(M,N,0,0.1,0,"rect","M rect tau0.1"))
        results.append(run(M,"gt",0,0,0,"rect","M rect GT"))
    elif stage=="evalL":
        results.append(evaluate([d for i in ids for d in L.get(i,[])],"L dense"))
        results.append(run(L,N,4,0.05,1,"tile","L G4 tau0.05 halo1"))
        results.append(run(L,N,8,0.05,1,"tile","L G8 tau0.05 halo1"))
        results.append(run(L,N,0,0.05,0,"rect","L rect tau0.05"))
        results.append(run(L,N,0,0.1,0,"rect","L rect tau0.1"))
        results.append(run(L,"gt",0,0,0,"rect","L rect GT"))
    elif stage=="topk":
        for G,k in ((4,10),(4,12),(8,40),(8,48)):
            results.append(run(M,N,G,0.05,0,"tile",f"M G{G} top{k}",topk=k))
    json.dump(results,open(f"occ_{stage}.json","w"),indent=1)
