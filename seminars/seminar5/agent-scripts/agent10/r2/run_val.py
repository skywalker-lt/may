# Run one ONNX on a fixed random subset of val2017 (CPU, one thread); write COCO detections, and for the fp32 model the
# per-image clipping statistics of every quantised activation tensor against its calibrated amax.
import sys, json, time, numpy as np, onnx, random
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent10/r2"); from pre import *; from qsel import quant_convs
model, tag, N = sys.argv[1], sys.argv[2], int(sys.argv[3]); taps = len(sys.argv) > 4 and sys.argv[4] == "taps"
OUT = "/data/tmp/ds-yolo/seminar5/work/agent10/r2"
files = sorted(os.listdir(VAL)); random.Random(1).shuffle(files); files = files[:N]
if taps:
    A = json.load(open(OUT + "/amax.json")); m = onnx.load(model); g = m.graph
    tens = sorted({n.input[0] for n in quant_convs(g, "all")}); del g.output[1:]
    for t in tens: g.output.append(onnx.helper.make_tensor_value_info(t, onnx.TensorProto.FLOAT, None))
    onnx.save(m, OUT + f"/_taps_{tag}.onnx"); model = OUT + f"/_taps_{tag}.onnx"
s = session(model); names = [o.name for o in s.get_outputs()]
dets = []; F = []; M = []; ids = []; t0 = time.time()
for k, f in enumerate(files):
    x, r, pw, ph = letterbox(f"{VAL}/{f}"); o = s.run(names, {"images": x}); img = int(f.split(".")[0]); ids.append(img)
    dets += to_coco(o[0], img, r, pw, ph)
    if taps:
        F.append([float((np.abs(a) > A[t]["entropy"]).mean()) for t, a in zip(names[1:], o[1:])])
        M.append([float(np.abs(a).max() / A[t]["entropy"]) for t, a in zip(names[1:], o[1:])])
    if k % 50 == 0: print(tag, k, "%.0fs" % (time.time() - t0), flush=True)
json.dump(dets, open(OUT + f"/dets_{tag}.json", "w"))
if taps: np.savez(OUT + f"/taps_{tag}.npz", ids=np.array(ids), frac=np.array(F), maxr=np.array(M), names=np.array(names[1:]))
json.dump(ids, open(OUT + f"/ids_{tag}.json", "w")); print(tag, "done %.0fs" % (time.time() - t0))
