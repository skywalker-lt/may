# Entropy (KL, TensorRT "EntropyCalibration2"-style: 2048-bin |x| histogram, 128 quantised bins) calibration of every
# activation tensor feeding a quantised Conv, on N train2017 images; CPU, onnxruntime, one thread. Two passes (max, histogram).
import sys, json, time, numpy as np, onnx, random
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent10/r2")
from pre import *; from qsel import quant_convs
N = int(sys.argv[1]) if len(sys.argv) > 1 else 128
OUT = "/data/tmp/ds-yolo/seminar5/work/agent10/r2"
m = onnx.load(ONNX); g = m.graph
tens = sorted({n.input[0] for n in quant_convs(g, "all")})
print("quantised convs", len(quant_convs(g, "all")), "activation tensors", len(tens), flush=True)
del g.output[1:]
for t in tens: g.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
onnx.save(m, OUT + "/_aug.onnx"); s = session(OUT + "/_aug.onnx")
names = [o.name for o in s.get_outputs()][1:]
files = sorted(os.listdir(TRAIN)); random.Random(0).shuffle(files); files = files[:N]
json.dump(files, open(OUT + "/calib_images.json", "w"))
mx = {t: 0.0 for t in names}; t0 = time.time()
for k, f in enumerate(files):
    o = s.run(names, {"images": letterbox(f"{TRAIN}/{f}")[0]})
    for t, a in zip(names, o): mx[t] = max(mx[t], float(np.abs(a).max()))
    if k % 32 == 0: print("pass1", k, "%.0fs" % (time.time() - t0), flush=True)
NB = 2048; hist = {t: np.zeros(NB, np.int64) for t in names}
for k, f in enumerate(files):
    o = s.run(names, {"images": letterbox(f"{TRAIN}/{f}")[0]})
    for t, a in zip(names, o): hist[t] += np.histogram(np.abs(a), bins=NB, range=(0, mx[t]))[0]
    if k % 32 == 0: print("pass2", k, "%.0fs" % (time.time() - t0), flush=True)
def kl_amax(h, top, nq=128):
    h = h.astype(np.float64); best, bi = np.inf, len(h)
    for i in range(nq, len(h) + 1):
        ref = h[:i].copy(); ref[i - 1] += h[i:].sum()
        idx = (np.arange(i) * nq // i)                       # quantised bin of each fine bin
        nz = h[:i] > 0
        cnt = np.bincount(idx[nz], minlength=nq).astype(np.float64)
        tot = np.bincount(idx, weights=h[:i], minlength=nq)
        q = np.where(nz, tot[idx] / np.maximum(cnt[idx], 1), 0.0)
        p = ref / ref.sum(); qs = q.sum()
        if qs <= 0: continue
        q = q / qs; mask = p > 0
        kl = np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12)))
        if kl < best: best, bi = kl, i
    return (bi + 0.5) * top / len(h)
amax = {}
for t in names:
    amax[t] = {"max": mx[t], "entropy": kl_amax(hist[t], mx[t])}
    c = np.cumsum(hist[t]) / hist[t].sum(); amax[t]["p99.99"] = float((np.searchsorted(c, 0.9999) + 1) * mx[t] / NB)
json.dump(amax, open(OUT + "/amax.json", "w"), indent=0)
r = np.array([amax[t]["entropy"] / amax[t]["max"] for t in names])
print("done %.0fs; entropy amax / max: median %.3f, min %.3f, max %.3f" % (time.time() - t0, np.median(r), r.min(), r.max()))
