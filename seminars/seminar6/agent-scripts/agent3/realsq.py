"""Square centred letterbox (SxS canvas, as the CLI) of YOLO26-L at S on the same 500 subset, same CPU fp32 pipeline as
realcrop.py, to isolate rect-letterbox vs square within one pipeline (agent 5's assumption)."""
import os,sys,json,time; os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"/data/YOLO-Master"); sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import numpy as np,cv2,torch; torch.set_num_threads(1)
from ultralytics import YOLO
import croproute as C
S=int(sys.argv[1]); d=json.load(open(sys.argv[2])); sub=d['sub']; cat_ids=sorted(C.coco.getCatIds())
net=YOLO("/data/yolo-quant-work/weights/yolo26l.pt").model.float().eval()
out=[];t0=time.time()
for k,i in enumerate(sub):
    img=cv2.imread(f"/data/datasets/coco/images/val2017/{i:012d}.jpg"); h,w=img.shape[:2]; r=S/max(w,h)
    nw,nh=round(w*r),round(h*r); x=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
    px=(S-nw)//2; py=(S-nh)//2; canvas=np.full((S,S,3),114,np.uint8); canvas[py:py+nh,px:px+nw]=x
    t=torch.from_numpy(canvas[:,:,::-1].copy()).permute(2,0,1)[None].float()/255
    with torch.no_grad(): y=net(t)
    y=(y[0] if isinstance(y,(list,tuple)) else y)[0].numpy()
    for x0,y0,x1,y1,s,c in y:
        if s<0.001: continue
        x0=(x0-px)/r; x1=(x1-px)/r; y0=(y0-py)/r; y1=(y1-py)/r
        out.append(dict(image_id=i,category_id=cat_ids[int(c)],bbox=[round(float(x0),2),round(float(y0),2),round(float(x1-x0),2),round(float(y1-y0),2)],score=round(float(s),5)))
    if k%50==0: print(k,f"{time.time()-t0:.0f}s",flush=True)
json.dump(out,open(f"realsq_L{S}_500.json","w"))
import contextlib,io
from pycocotools.cocoeval import COCOeval
with contextlib.redirect_stdout(io.StringIO()):
    E=COCOeval(C.coco,C.coco.loadRes(out),"bbox"); E.params.imgIds=sub; E.evaluate(); E.accumulate(); E.summarize()
print(f"SUBSET square centred letterbox L@{S} (CPU fp32): AP={E.stats[0]:.4f} AP50={E.stats[1]:.4f} AP75={E.stats[2]:.4f} S/M/L={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f} n={len(out)}",flush=True)
