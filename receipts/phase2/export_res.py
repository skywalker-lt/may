"""Export the public YOLO26-M at 512 and 768 (same exporter and settings as the Phase 1 graphs)."""
import shutil
from ultralytics import YOLO
for sz in (512, 768):
    f = YOLO("/data/yolo-quant-work/weights/yolo26m.pt").export(format="onnx", imgsz=sz, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
    shutil.move(f, f"/data/tmp/ds-yolo/phase2/onnx/yolo26m_{sz}.onnx"); print("exported", sz, flush=True)
