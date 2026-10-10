"""YOLO26-N at 320 on the CPU (one thread) as the thumbnail router: writes a COCO-format dump at conf 0.001.
fp32 on the CPU; AP device-independent to ~0.0004 (seminar convention)."""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,"/data/YOLO-Master")
import torch; torch.set_num_threads(1)
from ultralytics import YOLO
from pycocotools.coco import COCO
imgsz=int(sys.argv[1]) if len(sys.argv)>1 else 320
limit=int(sys.argv[2]) if len(sys.argv)>2 else 0
coco=COCO("/data/tmp/ds-yolo/seminar6/inputs/dumps/instances_val2017.json")
cat_ids=sorted(coco.getCatIds())  # 80 ids -> coco category ids
ids=sorted(coco.getImgIds())
if limit: ids=ids[:limit]
m=YOLO("/data/yolo-quant-work/weights/yolo26n.pt")
out=[]; t=time.time()
paths=[f"/data/datasets/coco/images/val2017/{i:012d}.jpg" for i in ids]
B=16
for b in range(0,len(paths),B):
    res=m.predict(paths[b:b+B],imgsz=imgsz,conf=0.001,max_det=300,device="cpu",verbose=False,half=False)
    for i,r in zip(ids[b:b+B],res):
        bx=r.boxes
        if bx is None or len(bx)==0: continue
        xyxy=bx.xyxy.numpy(); sc=bx.conf.numpy(); cl=bx.cls.numpy().astype(int)
        for (x0,y0,x1,y1),s,c in zip(xyxy,sc,cl):
            out.append(dict(image_id=int(i),category_id=int(cat_ids[c]),bbox=[round(float(x0),3),round(float(y0),3),round(float(x1-x0),3),round(float(y1-y0),3)],score=round(float(s),5)))
    if (b//B)%25==0: print(f"{b+B}/{len(paths)} {time.time()-t:.0f}s",flush=True)
json.dump(out,open(f"dump_yolo26n_{imgsz}_cpu{'_'+str(limit) if limit else ''}.json","w"))
print("done",len(out),f"{time.time()-t:.0f}s")
