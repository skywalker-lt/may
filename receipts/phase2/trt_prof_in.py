#!/usr/bin/env python3
"""Per-layer profile of an engine for a given real input; sums the time by branch prefix (e0_, e1_, ...) and the rest."""
import re, sys, collections
import numpy as np, tensorrt as trt
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart
def ck(r):
    err, *rest = r if isinstance(r, tuple) else (r,)
    assert err == cudart.cudaError_t.cudaSuccess, err
    return rest[0] if len(rest) == 1 else rest
eng_path, inputs = sys.argv[1], sys.argv[2:]
rt = trt.Runtime(trt.Logger(trt.Logger.ERROR)); eng = rt.deserialize_cuda_engine(open(eng_path, "rb").read()); ctx = eng.create_execution_context()
bufs, inp = [], None
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i); nbytes = int(np.prod(ctx.get_tensor_shape(n))) * np.dtype(trt.nptype(eng.get_tensor_dtype(n))).itemsize
    ptr = ck(cudart.cudaMalloc(nbytes)); ctx.set_tensor_address(n, ptr); bufs.append(ptr)
    if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT: inp = (ptr, nbytes)
class P(trt.IProfiler):
    def __init__(self): super().__init__(); self.t = collections.defaultdict(float)
    def report_layer_time(self, name, ms): self.t[name] += ms
for f in inputs:
    x = np.ascontiguousarray(np.load(f).astype(np.float32)); ck(cudart.cudaMemcpy(inp[0], x.ctypes.data, inp[1], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    p = P(); ctx.profiler = p
    for _ in range(50): ctx.execute_v2(bufs)
    by = collections.defaultdict(float)
    for k, v in p.t.items():
        m = re.search(r"\be(\d)_", k); by["branch e" + m.group(1) if m else "stem/router/other"] += v / 50
    print(f"PROFIN {eng_path.split('/')[-1].split('-')[0]} input={f.split('/')[-1]} total={sum(by.values()):.3f}ms " + " ".join(f"{k}={v:.3f}" for k, v in sorted(by.items())))
