#!/usr/bin/env python3
"""TensorRT 10.16 builder for the agent-10 precision receipt (H200 for AP, T4 / L4 for latency; never compared across devices).

Modes
  fp16      plain fp16 engine (the comparator, same builder, same session).
  explicit  ONNX already carries Q/DQ (make_qdq.py / make_srp.py): FP16 + INT8 flags, no calibrator. RECOMMENDED: the
            quantised graph and every scale are fixed in the file, so the H200, the T4 and the CPU reference
            (onnxruntime fake-quant) execute the same quantised function.
  implicit  plain ONNX + TensorRT entropy calibration (IInt8EntropyCalibrator2) from a directory of images; layer 0,
            the PSA blocks and the head's final convs pinned to fp16 (--scope stem also pins layers 6-23). Writes the
            calibration cache; `--cache2amax` turns that cache into the amax.json that make_qdq.py consumes, so the
            calibration can be done once on the H200 and frozen into an explicit Q/DQ ONNX.

Usage
  python build_int8.py fp16      yolo26m.onnx yolo26m_fp16.engine
  python build_int8.py explicit  yolo26m_q8_all.onnx yolo26m_q8_all.engine
  python build_int8.py implicit  yolo26m.onnx yolo26m_i8.engine --calib-dir /data/datasets/coco/images/train2017 \
         --n 1000 --cache yolo26m_entropy.cache [--scope all|stem]
  python build_int8.py cache2amax yolo26m_entropy.cache amax_trt.json
Every build also writes <engine>.layers.json (engine inspector, DETAILED) and prints the count of INT8 layers.
"""
import os, sys, re, json, random, argparse
import numpy as np
import tensorrt as trt
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart
import cv2

LOG = trt.Logger(trt.Logger.WARNING)

def letterbox(path, s=640):
    im = cv2.imread(path); h, w = im.shape[:2]; r = min(s / h, s / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    if (nh, nw) != (h, w): im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dh, dw = (s - nh) / 2, (s - nw) / 2
    top, bot = int(round(dh - 0.1)), int(round(dh + 0.1)); lef, rig = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bot, lef, rig, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0), r, lef, top

def ck(ret):
    err = ret[0] if isinstance(ret, tuple) else ret
    if int(err) != 0: raise RuntimeError(f"CUDA error {err}")
    return ret[1] if isinstance(ret, tuple) and len(ret) > 1 else None

class EntropyCalib(trt.IInt8EntropyCalibrator2):
    def __init__(self, files, cache, shape=(1, 3, 640, 640)):
        super().__init__(); self.files, self.cache, self.i = files, cache, 0
        self.nbytes = int(np.prod(shape)) * 4; self.d = ck(cudart.cudaMalloc(self.nbytes))
    def get_batch_size(self): return 1
    def get_batch(self, names):
        if self.i >= len(self.files): return None
        x = letterbox(self.files[self.i])[0]; self.i += 1
        ck(cudart.cudaMemcpy(self.d, x.ctypes.data, self.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        if self.i % 100 == 0: print("calibration batch", self.i, flush=True)
        return [int(self.d)]
    def read_calibration_cache(self):
        return open(self.cache, "rb").read() if os.path.exists(self.cache) else None
    def write_calibration_cache(self, c): open(self.cache, "wb").write(bytes(c))

def keep_fp(name, scope):
    m = re.match(r"/model\.(\d+)/", name); k = int(m.group(1)) if m else -1
    if k == 0: return True
    if name.startswith("/model.10/m/") or name.startswith("/model.22/m.0/m.0.1/"): return True
    if re.search(r"one2one_cv[23]\.\d/one2one_cv[23]\.\d\.2/Conv", name): return True
    if scope == "stem" and k >= 6: return True
    return False

def build(mode, onnx_path, engine_path, a):
    b = trt.Builder(LOG); net = b.create_network(0); p = trt.OnnxParser(net, LOG)
    if not p.parse_from_file(onnx_path):
        raise SystemExit("PARSE FAILED: " + "; ".join(str(p.get_error(i)) for i in range(p.num_errors)))
    cfg = b.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if mode in ("explicit", "implicit"): cfg.set_flag(trt.BuilderFlag.INT8)
    if mode == "implicit":
        files = sorted(os.listdir(a.calib_dir)); random.Random(0).shuffle(files)
        cfg.int8_calibrator = EntropyCalib([os.path.join(a.calib_dir, f) for f in files[:a.n]], a.cache)
        pinned = 0
        for i in range(net.num_layers):
            L = net.get_layer(i)
            if keep_fp(L.name, a.scope):
                L.precision = trt.float16; pinned += 1
                for j in range(L.num_outputs): L.set_output_type(j, trt.float16)
        cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        print("pinned to fp16:", pinned, "layers")
    ser = b.build_serialized_network(net, cfg)
    if ser is None: raise SystemExit("BUILD FAILED (see TensorRT log above)")
    open(engine_path, "wb").write(ser)
    eng = trt.Runtime(LOG).deserialize_cuda_engine(ser)
    info = eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    open(engine_path + ".layers.json", "w").write(info)
    n8 = info.count('"Int8"') + info.count("Int8")  # rough; the JSON lists per-layer input/output formats
    print(f"built {engine_path}; engine layers mentioning Int8: {info.count('Int8')}")

def cache2amax(cache, out):
    """TensorRT calibration cache: header line, then 'tensor: <hex big-endian float32 scale>'; amax = scale * 127."""
    import struct
    d = {}
    for line in open(cache, "rb").read().decode().splitlines()[1:]:
        if ":" not in line: continue
        k, v = line.rsplit(":", 1); s = struct.unpack("!f", bytes.fromhex(v.strip()))[0]
        d[k.strip()] = {"entropy": s * 127.0}
    json.dump(d, open(out, "w"), indent=0); print(len(d), "tensors ->", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode"); ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--calib-dir"); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--cache", default="calib.cache")
    ap.add_argument("--scope", default="all"); a = ap.parse_args()
    if a.mode == "cache2amax": cache2amax(a.src, a.dst)
    else: build(a.mode, a.src, a.dst, a)
