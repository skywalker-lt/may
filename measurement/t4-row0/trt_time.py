#!/usr/bin/env python3
"""trtexec substitute: time a serialized TensorRT engine with CUDA events (no host transfers) and optionally dump a per-layer profile."""
import argparse, ctypes, os, sys, time
import numpy as np, tensorrt as trt
try:
    from cuda.bindings import runtime as cudart  # cuda-python >= 12.6 layout
except ImportError:
    from cuda import cudart  # older cuda-python
ap = argparse.ArgumentParser(); ap.add_argument("engine"); ap.add_argument("--iters", type=int, default=500); ap.add_argument("--warm", type=int, default=200); ap.add_argument("--profile", default=None)
a = ap.parse_args()
def ck(r):
    if isinstance(r, tuple): err, *rest = r
    else: err, rest = r, []
    assert err == cudart.cudaError_t.cudaSuccess, err
    return rest[0] if len(rest) == 1 else rest
logger = trt.Logger(trt.Logger.ERROR); rt = trt.Runtime(logger)
eng = rt.deserialize_cuda_engine(open(a.engine, "rb").read()); ctx = eng.create_execution_context()
bufs = []
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i); shp = ctx.get_tensor_shape(n); dt = np.dtype(trt.nptype(eng.get_tensor_dtype(n)))
    nbytes = int(np.prod(shp)) * dt.itemsize; ptr = ck(cudart.cudaMalloc(nbytes)); ctx.set_tensor_address(n, ptr); bufs.append(ptr)
stream = ck(cudart.cudaStreamCreate()); s, e = ck(cudart.cudaEventCreate()), ck(cudart.cudaEventCreate())
for _ in range(a.warm): ctx.execute_async_v3(stream)
ck(cudart.cudaStreamSynchronize(stream))
ts = []
for _ in range(a.iters):
    ck(cudart.cudaEventRecord(s, stream)); ctx.execute_async_v3(stream); ck(cudart.cudaEventRecord(e, stream)); ck(cudart.cudaEventSynchronize(e))
    ts.append(ck(cudart.cudaEventElapsedTime(s, e)))
ts = np.array(ts)
print(f"TRTTIME {os.path.basename(a.engine).split('-')[0]} median={np.median(ts):.3f}ms mean={ts.mean():.3f} p90={np.percentile(ts,90):.3f} min={ts.min():.3f} iters={a.iters} trt={trt.__version__}")
if a.profile:
    class P(trt.IProfiler):
        def __init__(self): super().__init__(); self.t = {}
        def report_layer_time(self, name, ms): self.t[name] = self.t.get(name, 0.0) + ms
    p = P(); ctx.profiler = p
    for _ in range(50): ctx.execute_v2(bufs)
    tot = sum(p.t.values()) / 50
    with open(a.profile, "w") as f:
        f.write(f"# per-layer avg ms over 50 runs, total {tot:.3f} ms\n")
        for k, v in sorted(p.t.items(), key=lambda kv: -kv[1]): f.write(f"{v/50:8.4f}  {k}\n")
    print(f"PROFILE total={tot:.3f}ms layers={len(p.t)} -> {a.profile}")
