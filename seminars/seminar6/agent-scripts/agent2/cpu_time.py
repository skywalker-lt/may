import sys, time, torch; sys.path.insert(0, "/data/YOLO-Master"); torch.set_num_threads(1)
from ultralytics import YOLO
for w in ["yolo26l","yolo26m"]:
    m = YOLO(f"/data/yolo-quant-work/weights/{w}.pt").model.eval().float()
    for sz in [640, 448]:
        x = torch.randn(1,3,sz,sz)
        with torch.no_grad():
            m(x); t=time.time(); 
            for _ in range(3): m(x)
        print(w, sz, "s/forward", round((time.time()-t)/3,2), flush=True)
