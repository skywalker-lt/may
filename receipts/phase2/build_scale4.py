"""M stem + s/m/l/x-width tails behind one nested If (timing engine; 1x1 adapters where the widths differ)."""
import copy
import numpy as np, onnx, torch
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
from ultralytics import YOLO
from ultralytics.nn.modules.moe.weight_bank import bank_modules
OUT = "/data/tmp/ds-yolo/phase2/onnx"
def out_of(g, prefix):
    last = None
    for n in g.node:
        if n.name.startswith(prefix): last = n.output[0]
    return last
def chans(model, tensor):
    for vi in list(model.graph.value_info) + list(model.graph.output):
        if vi.name == tensor: return vi.type.tensor_type.shape.dim[1].dim_value
    return None
models = {s: onnx.load(f"{OUT}/_{s}640.onnx") for s in ("m", "l")}
for s in ("s", "x"):
    import shutil
    f = YOLO(f"/data/yolo-quant-work/weights/yolo26{s}.pt").export(format="onnx", imgsz=640, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
    shutil.move(f, f"{OUT}/_{s}640.onnx"); models[s] = onnx.load(f"{OUT}/_{s}640.onnx")
for s, m in models.items(): models[s] = onnx.shape_inference.infer_shapes(m)
g0 = models["m"].graph; L4, L5 = out_of(g0, "/model.4/"), out_of(g0, "/model.5/")
cm4, cm5 = chans(models["m"], L4), chans(models["m"], L5); print("M stem channels L4/L5:", cm4, cm5)
is_stem = lambda n: any(n.name.startswith(f"/model.{k}/") for k in range(6))
stem_nodes = [n for n in g0.node if is_stem(n)]; init0 = {t.name: t for t in g0.initializer}
stem_inits = [init0[x] for x in {i for n in stem_nodes for i in n.input} if x in init0]
order = ["s", "m", "l", "x"]
def branch(i, s):
    m = models[s]; g = m.graph; pre = f"e{i}_"
    l4, l5 = out_of(g, "/model.4/"), out_of(g, "/model.5/"); c4, c5 = chans(m, l4), chans(m, l5)
    nodes = [copy.deepcopy(n) for n in g.node if not is_stem(n)]
    used = {inp for n in nodes for inp in n.input} & {t.name for t in g.initializer}
    inits, extra = [], []
    src = {}
    for name, cin, cout, tag in ((L4, cm4, c4, "a4"), (L5, cm5, c5, "a5")):
        if cin == cout: src[name] = name
        else:  # 1x1 adapter from the M stem width to this tail's width (random weights, timing only)
            w = NH.from_array((np.random.randn(cout, cin, 1, 1) * (cin ** -0.5)).astype(np.float32), f"{pre}{tag}_w")
            inits.append(w); extra.append(H.make_node("Conv", [name, w.name], [f"{pre}{tag}"], kernel_shape=[1, 1], name=f"{pre}{tag}/Conv")); src[name] = f"{pre}{tag}"
    ren = lambda t: src[L4] if t == l4 else src[L5] if t == l5 else t if t == "" else pre + t
    for n in nodes:
        n.name = pre + n.name; n.input[:] = [ren(t) for t in n.input]; n.output[:] = [ren(t) for t in n.output]
    for t in g.initializer:
        if t.name in used:
            t2 = copy.deepcopy(t); t2.name = pre + t.name; inits.append(t2)
    return H.make_graph(extra + nodes, f"tail_{s}", [], [H.make_tensor_value_info(pre + g.output[0].name, TP.FLOAT, [1, 300, 6])], initializer=inits)
def nest(i):
    then_g = branch(i, order[i])
    if i == len(order) - 2: else_g = branch(len(order) - 1, order[-1])
    else:
        inner, inner_out = nest(i + 1); else_g = H.make_graph([inner], f"else{i}", [], [H.make_tensor_value_info(inner_out, TP.FLOAT, [1, 300, 6])])
    return H.make_node("If", [f"r_is{i}"], [f"sel{i}"], then_branch=then_g, else_branch=else_g, name=f"If_{i}"), f"sel{i}"
wb = YOLO("/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt").model.float().eval(); r = bank_modules(wb)[0].router; n = r.norm
c = lambda name, a: NH.from_array(np.asarray(a, dtype=np.float32) if np.asarray(a).dtype.kind == "f" else np.asarray(a), name)
r_inits = [c("r_mean", n.running_mean.numpy()[None]), c("r_rstd", torch.rsqrt(n.running_var + n.eps).numpy()[None]), c("r_w1", r.fc1.weight.detach().numpy()), c("r_b1", r.fc1.bias.detach().numpy()),
           c("r_w2", r.fc2.weight.detach().numpy()), c("r_b2", r.fc2.bias.detach().numpy()), NH.from_array(np.array([2, 3], dtype=np.int64), "r_axes")]
r_nodes = [H.make_node("ReduceMean", [L5, "r_axes"], ["r_pool"], keepdims=0, name="router/pool"), H.make_node("Sub", ["r_pool", "r_mean"], ["r_c"], name="router/sub"), H.make_node("Mul", ["r_c", "r_rstd"], ["r_n"], name="router/mul"),
           H.make_node("Gemm", ["r_n", "r_w1", "r_b1"], ["r_h"], transB=1, name="router/fc1"), H.make_node("Sigmoid", ["r_h"], ["r_s"], name="router/sig"), H.make_node("Mul", ["r_h", "r_s"], ["r_a"], name="router/silu"),
           H.make_node("Gemm", ["r_a", "r_w2", "r_b2"], ["r_logits"], transB=1, name="router/fc2"), H.make_node("ArgMax", ["r_logits"], ["r_idx"], axis=1, keepdims=0, name="router/argmax"), H.make_node("Squeeze", ["r_idx"], ["r_i"], name="router/sq")]
for i in range(3):
    r_inits.append(NH.from_array(np.array(i, dtype=np.int64), f"r_k{i}")); r_nodes.append(H.make_node("Equal", ["r_i", f"r_k{i}"], [f"r_is{i}"], name=f"router/is{i}"))
if_node, sel = nest(0)
graph = H.make_graph(stem_nodes + r_nodes + [if_node, H.make_node("Identity", [sel], ["output0"], name="out")], "if_scale4", list(g0.input), [H.make_tensor_value_info("output0", TP.FLOAT, [1, 300, 6])], initializer=stem_inits + r_inits)
model = H.make_model(graph, opset_imports=list(models["m"].opset_import), ir_version=models["m"].ir_version); onnx.save(model, f"{OUT}/if_scale4.onnx"); onnx.checker.check_model(f"{OUT}/if_scale4.onnx")
print("if_scale4 ok", round(len(model.SerializeToString()) / 1e6, 1), "MB; branch order s,m,l,x = router index 0..3")
