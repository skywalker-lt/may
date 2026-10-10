# Insert explicit INT8 Q/DQ (symmetric, activation per-tensor from amax.json, weight per-output-channel max) into the
# YOLO26-M ONNX for a scope: all | stem (layers 1-5) | tail (layers 6-23). Layer 0, the PSA blocks and the head's final
# convs stay floating point (qsel.keep_fp). The result is the TensorRT explicit-quantisation form (QuantizeLinear /
# DequantizeLinear, int8, zero point 0); onnxruntime runs it as exact fake-quant on the CPU.
import os, sys, json, numpy as np, onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from qsel import quant_convs
def make(src, amax_json, scope, dst, method="entropy"):
    m = onnx.load(src); g = m.graph; A = json.load(open(amax_json)); inits = {t.name: t for t in g.initializer}
    qn = {n.name for n in quant_convs(g, scope)}; done = set(); new = []; extra = []
    for n in g.node:
        if n.name in qn:
            a = n.input[0]
            if a not in done:
                s = max(A[a][method], 1e-8) / 127.0
                extra += [NH.from_array(np.array(s, np.float32), a + "_qs"), NH.from_array(np.array(0, np.int8), a + "_qz")]
                new += [H.make_node("QuantizeLinear", [a, a + "_qs", a + "_qz"], [a + "_q"], name=a + "_Q"),
                        H.make_node("DequantizeLinear", [a + "_q", a + "_qs", a + "_qz"], [a + "_dq"], name=a + "_DQ")]
                done.add(a)
            w = n.input[1]; W = NH.to_array(inits[w]); sw = np.maximum(np.abs(W).reshape(W.shape[0], -1).max(1), 1e-8) / 127.0
            extra += [NH.from_array(sw.astype(np.float32), w + "_ws"), NH.from_array(np.zeros(W.shape[0], np.int8), w + "_wz")]
            new += [H.make_node("QuantizeLinear", [w, w + "_ws", w + "_wz"], [w + "_wq"], axis=0, name=w + "_Q"),
                    H.make_node("DequantizeLinear", [w + "_wq", w + "_ws", w + "_wz"], [w + "_wdq"], axis=0, name=w + "_DQ")]
            n.input[0] = a + "_dq"; n.input[1] = w + "_wdq"
        new.append(n)
    del g.node[:]; g.node.extend(new); g.initializer.extend(extra)
    onnx.checker.check_model(m); onnx.save(m, dst)
    print(f"{dst}: {len(qn)} convs quantised, {len(done)} activation tensors")
if __name__ == "__main__":
    make(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "entropy")
