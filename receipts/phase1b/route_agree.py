#!/usr/bin/env python3
"""Route agreement on the T4: router logits from the full conditional fp16 engine (extra output r_logits) against an
fp32 engine of the router-only graph, over val2017 with the Ultralytics letterbox (640, pad 114)."""
import glob, sys, collections
import cv2, numpy as np, tensorrt as trt
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart
def ck(r):
    err, *rest = r if isinstance(r, tuple) else (r,)
    assert err == cudart.cudaError_t.cudaSuccess, err
    return rest[0] if len(rest) == 1 else rest
lg = trt.Logger(trt.Logger.ERROR)
def build(onnx_path, fp16):
    b = trt.Builder(lg); net = b.create_network(0); p = trt.OnnxParser(net, lg)
    assert p.parse_from_file(onnx_path), [p.get_error(i) for i in range(p.num_errors)]
    cfg = b.create_builder_config()
    if fp16: cfg.set_flag(trt.BuilderFlag.FP16)
    return trt.Runtime(lg).deserialize_cuda_engine(b.build_serialized_network(net, cfg))
class Run:
    def __init__(self, eng):
        self.eng, self.ctx, self.io = eng, eng.create_execution_context(), {}
        for i in range(eng.num_io_tensors):
            n = eng.get_tensor_name(i); shp = tuple(self.ctx.get_tensor_shape(n)); dt = np.dtype(trt.nptype(eng.get_tensor_dtype(n)))
            ptr = ck(cudart.cudaMalloc(int(np.prod(shp)) * dt.itemsize)); self.ctx.set_tensor_address(n, ptr); self.io[n] = (ptr, shp, dt)
        self.stream = ck(cudart.cudaStreamCreate())
    def __call__(self, x, out):
        ptr, shp, dt = self.io["images"]; ck(cudart.cudaMemcpy(ptr, x.ctypes.data, x.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
        self.ctx.execute_async_v3(self.stream); ck(cudart.cudaStreamSynchronize(self.stream))
        ptr, shp, dt = self.io[out]; y = np.empty(shp, dt); ck(cudart.cudaMemcpy(y.ctypes.data, ptr, y.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)); return y.astype(np.float32)
def letterbox(im, s=640):
    h, w = im.shape[:2]; r = min(s / h, s / w); nw, nh = round(w * r), round(h * r)
    if (w, h) != (nw, nh): im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    dw, dh = (s - nw) / 2, (s - nh) / 2
    im = cv2.copyMakeBorder(im, round(dh - 0.1), round(dh + 0.1), round(dw - 0.1), round(dw + 0.1), cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255)
full16 = Run(build(sys.argv[1], True)); ref32 = Run(build(sys.argv[2], False)); ro16 = Run(build(sys.argv[2], True))
files = sorted(glob.glob("/root/coco/images/val2017/*.jpg")); n = int(sys.argv[3]) if len(sys.argv) > 3 else len(files)
A, B, C = [], [], []
for f in files[:n]:
    x = letterbox(cv2.imread(f)); A.append(full16(x, "r_logits")[0]); B.append(ref32(x, "r_logits")[0]); C.append(ro16(x, "r_logits")[0])
A, B, C = np.array(A), np.array(B), np.array(C)
srt = np.sort(B, 1); margin = srt[:, -1] - srt[:, -2]
print(f"ROUTE images={len(A)} | full fp16 engine vs fp32 router: argmax agreement {np.mean(A.argmax(1) == B.argmax(1)):.4f} ({int(np.sum(A.argmax(1) != B.argmax(1)))} flips), max |dlogit| {np.abs(A - B).max():.4f}, mean {np.abs(A - B).mean():.5f}")
print(f"ROUTE router-only fp16 vs fp32: agreement {np.mean(C.argmax(1) == B.argmax(1)):.4f} ({int(np.sum(C.argmax(1) != B.argmax(1)))} flips)")
print(f"ROUTE fp32 margin top1-top2: median {np.median(margin):.3f}; share below 0.01: {np.mean(margin < 0.01):.4f}; below 0.05: {np.mean(margin < 0.05):.4f}; margins of flipped images: {np.round(np.sort(margin[A.argmax(1) != B.argmax(1)])[:12], 4).tolist()}")
print(f"ROUTE branch histogram fp16 engine {dict(sorted(collections.Counter(A.argmax(1).tolist()).items()))} | fp32 {dict(sorted(collections.Counter(B.argmax(1).tolist()).items()))}")
