#!/usr/bin/env python3
"""Run a static batch-1 TensorRT engine over val2017 and write COCO detections (conf >= 0.001, the engine's own 300
end-to-end outputs), plus every scalar output (the SRP taps: srp_maxr, srp_frac, srp_score) per image.
Validate once against the CLI: the fp16 engine through this script must reproduce the CLI dump's 0.5261 within 0.0005.
Usage: python dump_trt.py engine val_dir out.json [--limit N]"""
import os, sys, json, argparse, numpy as np, tensorrt as trt
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from build_int8 import letterbox, ck, LOG
COCO80 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
ap = argparse.ArgumentParser(); ap.add_argument("engine"); ap.add_argument("val"); ap.add_argument("out"); ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()
eng = trt.Runtime(LOG).deserialize_cuda_engine(open(a.engine, "rb").read()); ctx = eng.create_execution_context()
stream = ck(cudart.cudaStreamCreate()); bufs = {}
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i); shp = tuple(eng.get_tensor_shape(n)); dt = trt.nptype(eng.get_tensor_dtype(n))
    h = np.zeros(shp if len(shp) else (1,), dt); d = ck(cudart.cudaMalloc(max(h.nbytes, 4)))
    bufs[n] = (h, d, eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT); ctx.set_tensor_address(n, int(d))
files = sorted(os.listdir(a.val)); files = files[:a.limit] if a.limit else files
dets, taps = [], {}
for f in files:
    x, r, pw, ph = letterbox(os.path.join(a.val, f)); img = int(f.split(".")[0])
    hin, din, _ = bufs["images"]; ck(cudart.cudaMemcpyAsync(din, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream))
    ctx.execute_async_v3(stream)
    for n, (h, d, isin) in bufs.items():
        if not isin: ck(cudart.cudaMemcpyAsync(h.ctypes.data, d, h.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream))
    ck(cudart.cudaStreamSynchronize(stream))
    out = bufs["output0"][0].reshape(-1, 6); out = out[out[:, 4] >= 0.001]
    for x1, y1, x2, y2, sc, c in out.astype(np.float64):
        x1 = (x1 - pw) / r; x2 = (x2 - pw) / r; y1 = (y1 - ph) / r; y2 = (y2 - ph) / r
        dets.append({"image_id": img, "category_id": COCO80[int(c)], "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(sc)})
    t = {n: float(h.reshape(-1)[0]) for n, (h, d, isin) in bufs.items() if not isin and n != "output0"}
    if t: taps[img] = t
json.dump(dets, open(a.out, "w"))
if taps: json.dump(taps, open(a.out.replace(".json", "_taps.json"), "w"))
print("wrote", a.out, len(dets), "detections", len(files), "images")
