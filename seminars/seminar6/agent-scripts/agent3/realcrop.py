"""Direct check of the filter approximation: YOLO26-L at long side S on the full image and on the router rectangle at the
same pixel density (own preprocessing: linear resize by r = S/max(w,h), pad right/bottom with 114 to multiples of 32).
Writes both dumps (COCO json) for a subset of val2017. CPU fp32, one thread."""
import os, sys, json, time, math
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"/data/YOLO-Master"); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch; torch.set_num_threads(1)
from ultralytics import YOLO
S=int(sys.argv[1]); n=int(sys.argv[2]); seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
import croproute as C   # loads COCO and helpers (no main)
N=C.load(C.D+"dump_yolo26n_coco.json"); R=C.rects(N,0.05)
rng=np.random.default_rng(seed); sub=sorted(rng.choice(C.ids,n,replace=False).tolist())
cat_ids=sorted(C.coco.getCatIds())
net=YOLO("/data/yolo-quant-work/weights/yolo26l.pt").model.float().eval()
def run(img,r,ox,oy):
    h,w=img.shape[:2]; nw,nh=max(1,round(w*r)),max(1,round(h*r))
    x=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
    pw,ph=C.c32(nw),C.c32(nh); canvas=np.full((ph,pw,3),114,np.uint8); canvas[:nh,:nw]=x
    t=torch.from_numpy(canvas[:,:,::-1].copy()).permute(2,0,1)[None].float()/255
    with torch.no_grad(): y=net(t)
    y=y[0] if isinstance(y,(list,tuple)) else y
    y=y[0].numpy(); out=[]
    for x0,y0,x1,y1,s,c in y:
        if s<0.001: continue
        x0=min(max(x0,0),nw)/r+ox; x1=min(max(x1,0),nw)/r+ox; y0=min(max(y0,0),nh)/r+oy; y1=min(max(y1,0),nh)/r+oy
        out.append([x0,y0,x1-x0,y1-y0,float(s),int(c)])
    return out,pw*ph
dense=[];crop=[];t0=time.time();px=[0,0]
for k,i in enumerate(sub):
    img=cv2.imread(f"/data/datasets/coco/images/val2017/{i:012d}.jpg"); h,w=img.shape[:2]; r=S/max(w,h)
    d,a=run(img,r,0,0); px[0]+=a
    dense+=[dict(image_id=i,category_id=cat_ids[c],bbox=[round(float(v),2) for v in b[:4]],score=round(s,5)) for *b,s,c in [(*q[:4],q[4],q[5]) for q in d]]
    if R[i] is not None:
        x0,y0,x1,y1=[int(round(v)) for v in R[i]]; x0=max(0,x0); y0=max(0,y0)
        c_,a=run(img[y0:y1,x0:x1],r,x0,y0); px[1]+=a
        crop+=[dict(image_id=i,category_id=cat_ids[c],bbox=[round(float(v),2) for v in b[:4]],score=round(s,5)) for *b,s,c in [(*q[:4],q[4],q[5]) for q in c_]]
    if k%10==0: print(k,f"{time.time()-t0:.0f}s",flush=True)
json.dump(dict(sub=sub,dense=dense,crop=crop,px=px),open(f"realcrop_L{S}_{n}_{seed}.json","w"))
print("done",f"{time.time()-t0:.0f}s","pixel ratio crop/dense",px[1]/px[0])
