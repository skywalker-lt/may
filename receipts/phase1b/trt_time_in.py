#!/usr/bin/env python3
"""trt_time.py with real inputs: the given .npy tensors are copied to the device input in turn (one per iteration,
outside the timed region), so each iteration can take a different branch. Prints the median per input and overall."""
import argparse, os
import numpy as np, tensorrt as trt
try:
    from cuda.bindings import runtime as cudart
except ImportError:
    from cuda import cudart
ap = argparse.ArgumentParser(); ap.add_argument("engine"); ap.add_argument("--inputs", nargs="+", required=True)
ap.add_argument("--iters", type=int, default=500); ap.add_argument("--warm", type=int, default=200); ap.add_argument("--tag", default="")
a = ap.parse_args()
def ck(r):
    err, *rest = r if isinstance(r, tuple) else (r,)
    assert err == cudart.cudaError_t.cudaSuccess, err
    return rest[0] if len(rest) == 1 else rest
rt = trt.Runtime(trt.Logger(trt.Logger.ERROR)); eng = rt.deserialize_cuda_engine(open(a.engine, "rb").read()); ctx = eng.create_execution_context()
inp = None
for i in range(eng.num_io_tensors):
    n = eng.get_tensor_name(i); nbytes = int(np.prod(ctx.get_tensor_shape(n))) * np.dtype(trt.nptype(eng.get_tensor_dtype(n))).itemsize
    ptr = ck(cudart.cudaMalloc(nbytes)); ctx.set_tensor_address(n, ptr)
    if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT: inp = (ptr, nbytes)
xs = [np.ascontiguousarray(np.load(f).astype(np.float32)) for f in a.inputs]
stream = ck(cudart.cudaStreamCreate()); s, e = ck(cudart.cudaEventCreate()), ck(cudart.cudaEventCreate())
def put(x): ck(cudart.cudaMemcpy(inp[0], x.ctypes.data, inp[1], cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
for i in range(a.warm): put(xs[i % len(xs)]); ctx.execute_async_v3(stream)
ck(cudart.cudaStreamSynchronize(stream))
ts = [[] for _ in xs]
for i in range(a.iters * len(xs)):
    j = i % len(xs); put(xs[j])
    ck(cudart.cudaEventRecord(s, stream)); ctx.execute_async_v3(stream); ck(cudart.cudaEventRecord(e, stream)); ck(cudart.cudaEventSynchronize(e))
    ts[j].append(ck(cudart.cudaEventElapsedTime(s, e)))
allt = np.concatenate([np.array(t) for t in ts])
per = " ".join(f"{os.path.basename(f).replace('.npy','')}={np.median(t):.3f}" for f, t in zip(a.inputs, ts))
print(f"TRTIN {a.tag or os.path.basename(a.engine).split('-')[0]} overall_median={np.median(allt):.3f}ms p90={np.percentile(allt, 90):.3f} | {per} | iters/input={a.iters}")
