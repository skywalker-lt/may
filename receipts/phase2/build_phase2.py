"""Phase 2 seminar receipts (local tool): resolution exports, stem-only graphs (router cost), and a depth-routed engine
(YOLO26-M stem + {M tail, L tail}; M and L share widths, L doubles the depth)."""
import copy, shutil
import numpy as np, onnx, torch
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
from onnx.utils import Extractor
from ultralytics import YOLO
from ultralytics.nn.modules.moe.weight_bank import bank_modules

OUT = "/data/tmp/ds-yolo/phase2/onnx"
def export(pt, sz, name):
    f = YOLO(pt).export(format="onnx", imgsz=sz, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
    shutil.move(f, f"{OUT}/{name}.onnx"); return onnx.load(f"{OUT}/{name}.onnx")
def out_of(g, prefix):
    last = None
    for n in g.node:
        if n.name.startswith(prefix): last = n.output[0]
    return last
# 1. resolution rows
export("/data/yolo-quant-work/weights/yolo26s.pt", 768, "yolo26s_768")
# 2. stem-only graphs: layers 0-5 of M at 640, of N at 640 and 320 (router cost = stem + pooling)
for pt, sz, name in (("/data/yolo-quant-work/weights/yolo26m.pt", 640, "stem_m_640"), ("/data/yolo-quant-work/weights/yolo26n.pt", 640, "stem_n_640"), ("/data/yolo-quant-work/weights/yolo26n.pt", 320, "stem_n_320")):
    m = export(pt, sz, f"_{name}_full"); L5 = out_of(m.graph, "/model.5/")
    sub = Extractor(m).extract_model(["images"], [L5])
    g = sub.graph; g.node.append(H.make_node("ReduceMean", [L5, "ax"], ["pooled"], keepdims=0, name="pool")); g.initializer.append(NH.from_array(np.array([2, 3], dtype=np.int64), "ax"))
    del g.output[:]; g.output.append(H.make_tensor_value_info("pooled", TP.FLOAT, None)); onnx.save(sub, f"{OUT}/{name}.onnx"); print(name, "ok", flush=True)
# 3. depth-routed engine: M stem, branches = M tail, L tail (router: the Phase 1 linear router restricted to two logits)
mM = export("/data/yolo-quant-work/weights/yolo26m.pt", 640, "_m640"); mL = export("/data/yolo-quant-work/weights/yolo26l.pt", 640, "_l640")
g0 = mM.graph; L4, L5 = out_of(g0, "/model.4/"), out_of(g0, "/model.5/")
assert out_of(mL.graph, "/model.5/") == L5 and out_of(mL.graph, "/model.4/") == L4
is_stem = lambda n: any(n.name.startswith(f"/model.{k}/") for k in range(6))
stem_nodes = [n for n in g0.node if is_stem(n)]; init0 = {t.name: t for t in g0.initializer}
stem_inits = [init0[x] for x in {i for n in stem_nodes for i in n.input} if x in init0]
def branch(model, i):
    g = model.graph; pre = f"e{i}_"; keep = {L4, L5}
    nodes = [copy.deepcopy(n) for n in g.node if not is_stem(n)]
    used = {inp for n in nodes for inp in n.input} & {t.name for t in g.initializer}
    ren = lambda s: s if (s in keep or s == "") else pre + s
    for n in nodes:
        n.name = pre + n.name; n.input[:] = [ren(s) for s in n.input]; n.output[:] = [ren(s) for s in n.output]
    inits = []
    for t in g.initializer:
        if t.name in used:
            t2 = copy.deepcopy(t); t2.name = pre + t.name; inits.append(t2)
    return H.make_graph(nodes, f"branch{i}", [], [H.make_tensor_value_info(pre + g.output[0].name, TP.FLOAT, [1, 300, 6])], initializer=inits)
wb = YOLO("/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt").model.float().eval(); r = bank_modules(wb)[0].router; n = r.norm
c = lambda name, a: NH.from_array(np.asarray(a, dtype=np.float32) if np.asarray(a).dtype.kind == "f" else np.asarray(a), name)
r_inits = [c("r_mean", n.running_mean.numpy()[None]), c("r_rstd", torch.rsqrt(n.running_var + n.eps).numpy()[None]), c("r_w1", r.fc1.weight.detach().numpy()), c("r_b1", r.fc1.bias.detach().numpy()),
           c("r_w2", r.fc2.weight.detach().numpy()[:2]), c("r_b2", r.fc2.bias.detach().numpy()[:2]), NH.from_array(np.array([2, 3], dtype=np.int64), "r_axes"), NH.from_array(np.array(0, dtype=np.int64), "r_k0")]
r_nodes = [H.make_node("ReduceMean", [L5, "r_axes"], ["r_pool"], keepdims=0, name="router/pool"), H.make_node("Sub", ["r_pool", "r_mean"], ["r_c"], name="router/sub"), H.make_node("Mul", ["r_c", "r_rstd"], ["r_n"], name="router/mul"),
           H.make_node("Gemm", ["r_n", "r_w1", "r_b1"], ["r_h"], transB=1, name="router/fc1"), H.make_node("Sigmoid", ["r_h"], ["r_s"], name="router/sig"), H.make_node("Mul", ["r_h", "r_s"], ["r_a"], name="router/silu"),
           H.make_node("Gemm", ["r_a", "r_w2", "r_b2"], ["r_logits"], transB=1, name="router/fc2"), H.make_node("ArgMax", ["r_logits"], ["r_idx"], axis=1, keepdims=0, name="router/argmax"),
           H.make_node("Squeeze", ["r_idx"], ["r_i"], name="router/sq"), H.make_node("Equal", ["r_i", "r_k0"], ["r_is0"], name="router/is0")]
if_node = H.make_node("If", ["r_is0"], ["sel"], then_branch=branch(mM, 0), else_branch=branch(mL, 1), name="If_0")
graph = H.make_graph(stem_nodes + r_nodes + [if_node, H.make_node("Identity", ["sel"], ["output0"], name="out")], "if_depth_ml", list(g0.input), [H.make_tensor_value_info("output0", TP.FLOAT, [1, 300, 6])], initializer=stem_inits + r_inits)
model = H.make_model(graph, opset_imports=list(mM.opset_import), ir_version=mM.ir_version); onnx.save(model, f"{OUT}/if_depth_ml.onnx"); onnx.checker.check_model(f"{OUT}/if_depth_ml.onnx")
print("if_depth_ml ok", round(len(model.SerializeToString()) / 1e6, 1), "MB", flush=True)
