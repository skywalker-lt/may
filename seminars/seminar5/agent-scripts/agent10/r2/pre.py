# Shared preprocessing / postprocessing for CPU (onnxruntime, one thread) runs of the YOLO26-M ONNX.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, cv2, onnxruntime as ort
ONNX = "/data/tmp/l4-row0/onnx/yolo26m.onnx"
VAL = "/data/datasets/coco/images/val2017"; TRAIN = "/data/datasets/coco/images/train2017"
COCO80 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
def letterbox(path, s=640):
    im = cv2.imread(path); h, w = im.shape[:2]; r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w): im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = (s - nh) / 2, (s - nw) / 2
    top, bot = int(round(dh - 0.1)), int(round(dh + 0.1)); lef, rig = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bot, lef, rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    x = im[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), r, lef, top
def session(path, extra_outputs=()):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
def to_coco(out, img_id, r, padw, padh, conf=0.001):
    d = out[0]; d = d[d[:, 4] >= conf]; res = []
    for x1, y1, x2, y2, sc, c in d:
        x1 = (x1 - padw) / r; x2 = (x2 - padw) / r; y1 = (y1 - padh) / r; y2 = (y2 - padh) / r
        res.append({"image_id": int(img_id), "category_id": COCO80[int(c)], "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": float(sc)})
    return res
