"""Agent 2, round 2: build the four LAND bypass configurations of public YOLO26-L (m[1] of each listed layer -> Identity).
C3k2 (C2f.forward): y=[a,b,m0(b),m1(m0)] -> with m1=Identity cv2 sees [a,b,m0,m0]. C2PSA layer 10: m=Sequential(PSA0,PSA1) -> PSA1 skipped.
Usage: PYTHONPATH=/data/YOLO-Master python surgery.py [--export]"""
import sys, os, shutil, torch, torch.nn as nn
torch.set_num_threads(1)
from ultralytics import YOLO
SRC = "/data/yolo-quant-work/weights/yolo26l.pt"
OUT = "/data/tmp/ds-yolo/seminar5/work/agent2/request"
META = "/data/tmp/l4-row0/onnx/yolo26m.metadata.yaml"
STEM = [2, 4]; P45 = [6, 8, 10, 13, 19]; P3 = [16]
CONFIGS = {
    "yolo26l_land_b0": STEM + P45 + P3,   # all eight second units bypassed (M topology)
    "yolo26l_land_b1": STEM + P45,        # deep P3 (unit 16 kept)
    "yolo26l_land_b2": STEM + P3,         # deep P4/P5
    "yolo26l_land_b3": STEM,              # deep everywhere except stem
}
def build(layers):
    y = YOLO(SRC)
    seq = y.model.model
    for i in layers:
        m = seq[i].m
        assert len(m) == 2, (i, len(m))
        m[1] = nn.Identity()
    return y
if __name__ == "__main__":
    for name, layers in CONFIGS.items():
        y = build(layers)
        npar = sum(p.numel() for p in y.model.parameters())
        pt = f"{OUT}/{name}.pt"
        y.save(pt)
        print(name, "bypassed", layers, "params", npar, flush=True)
        if "--export" in sys.argv:
            f = YOLO(pt).export(format="onnx", imgsz=640, batch=1, dynamic=False, half=False, simplify=True, device="cpu")
            dst = f"{OUT}/{name}.onnx"
            if os.path.abspath(f) != dst: shutil.move(f, dst)
            shutil.copy(META, f"{OUT}/{name}.metadata.yaml")
            print("exported", dst, os.path.getsize(dst), flush=True)
