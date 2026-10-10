"""Build the conditional top-1 graph (local tool): shared stem (layers 0-5), router, then a nested ONNX If whose four
branches are static copies of layers 6-23, one per expert. Only the taken branch executes in TensorRT."""
import copy, shutil, sys
import numpy as np, onnx, onnxruntime as ort, torch
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.moe.weight_bank import BankConv2d, bank_modules

OUT = "/data/tmp/ds-yolo/phase1/onnx"; WB = "/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt"
import cv2
from ultralytics.data.augment import LetterBox
im = LetterBox((640, 640), auto=False)(image=cv2.imread("/data/datasets/coco/images/val2017/000000000139.jpg"))
x = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]

y = YOLO(WB); wb = y.model.float().eval()
st = bank_modules(wb)[0].state; st.top_k = 1
router = bank_modules(wb)[0].router
with torch.no_grad():
    ref = wb(x)[0].numpy(); idx_ref = int(st.logits.argmax(1))
E = st.num_experts

def dense_expert(i):
    m = copy.deepcopy(wb)
    for blk in m.modules():
        if isinstance(blk, Conv) and isinstance(blk.conv, BankConv2d):
            b = blk.conv; c = nn.Conv2d(b.in_channels, b.out_channels, 1, bias=False)
            c.weight.data.copy_(b.experts[i].data); blk.conv = c
    m.yaml.pop("weight_bank", None)
    return m

paths = []
for i in range(E):
    yi = YOLO(WB); yi.model = dense_expert(i)
    f = yi.export(format="onnx", imgsz=640, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
    p = f"{OUT}/_expert{i}.onnx"; shutil.move(f, p); paths.append(p)

models = [onnx.load(p) for p in paths]
g0 = models[0].graph
def out_of(g, prefix):  # last tensor produced under a layer prefix, e.g. /model.5/
    last = None
    for n in g.node:
        if n.name.startswith(prefix): last = n.output[0]
    return last
L4, L5 = out_of(g0, "/model.4/"), out_of(g0, "/model.5/")
print("outer tensors:", L4, L5)
stem_nodes = [n for n in g0.node if any(n.name.startswith(f"/model.{k}/") for k in range(6))]
stem_out = {o for n in stem_nodes for o in n.output}
stem_init_names = {i for n in stem_nodes for i in n.input}
init0 = {t.name: t for t in g0.initializer}
stem_inits = [init0[n] for n in stem_init_names if n in init0]
for i, m in enumerate(models[1:], 1):  # the stem is identical in every expert export
    assert out_of(m.graph, "/model.5/") == L5

def branch(i):
    g = models[i].graph; pre = f"e{i}_"
    keep = {L4, L5}
    stem_names = {n.name for n in g.node if any(n.name.startswith(f"/model.{k}/") for k in range(6))}
    nodes = [copy.deepcopy(n) for n in g.node if n.name not in stem_names]
    used_init = {inp for n in nodes for inp in n.input} & {t.name for t in g.initializer}
    ren = lambda s: s if (s in keep or s == "") else pre + s
    for n in nodes:
        n.name = pre + n.name
        n.input[:] = [ren(s) for s in n.input]; n.output[:] = [ren(s) for s in n.output]
    inits = []
    for t in g.initializer:
        if t.name in used_init:
            t2 = copy.deepcopy(t); t2.name = pre + t.name; inits.append(t2)
    out = H.make_tensor_value_info(pre + g.output[0].name, TP.FLOAT, [1, 300, 6])
    return H.make_graph(nodes, f"expert{i}", [], [out], initializer=inits)

# router in the main graph (plain arithmetic standardisation, two Gemm, SiLU, ArgMax)
n = router.norm
c = lambda name, a: NH.from_array(np.asarray(a), name)
r_inits = [c("r_mean", n.running_mean.numpy()[None]), c("r_rstd", torch.rsqrt(n.running_var + n.eps).numpy()[None]),
           c("r_w1", router.fc1.weight.detach().numpy()), c("r_b1", router.fc1.bias.detach().numpy()),
           c("r_w2", router.fc2.weight.detach().numpy()), c("r_b2", router.fc2.bias.detach().numpy()), c("r_axes", np.array([2, 3], dtype=np.int64))]
r_nodes = [H.make_node("ReduceMean", [L5, "r_axes"], ["r_pool"], keepdims=0, name="router/pool"),
           H.make_node("Sub", ["r_pool", "r_mean"], ["r_c"], name="router/sub"), H.make_node("Mul", ["r_c", "r_rstd"], ["r_n"], name="router/mul"),
           H.make_node("Gemm", ["r_n", "r_w1", "r_b1"], ["r_h"], transB=1, name="router/fc1"),
           H.make_node("Sigmoid", ["r_h"], ["r_s"], name="router/sig"), H.make_node("Mul", ["r_h", "r_s"], ["r_a"], name="router/silu"),
           H.make_node("Gemm", ["r_a", "r_w2", "r_b2"], ["r_logits"], transB=1, name="router/fc2"),
           H.make_node("ArgMax", ["r_logits"], ["r_idx"], axis=1, keepdims=0, name="router/argmax"),
           H.make_node("Squeeze", ["r_idx"], ["r_i"], name="router/squeeze")]
for i in range(E - 1):
    r_inits.append(c(f"r_k{i}", np.array(i, dtype=np.int64)))
    r_nodes.append(H.make_node("Equal", ["r_i", f"r_k{i}"], [f"r_is{i}"], name=f"router/is{i}"))

def nest(i):  # If(is_i) then expert i else nest(i+1); the last else is expert E-1
    then_g = branch(i)
    if i == E - 2:
        else_g = branch(E - 1)
    else:
        inner, inner_out = nest(i + 1)
        else_g = H.make_graph([inner], f"else{i}", [], [H.make_tensor_value_info(inner_out, TP.FLOAT, [1, 300, 6])])
    out = f"sel{i}"
    return H.make_node("If", [f"r_is{i}"], [out], then_branch=then_g, else_branch=else_g, name=f"If_{i}"), out

if_node, sel = nest(0)
ident = H.make_node("Identity", [sel], ["output0"], name="out")
graph = H.make_graph(stem_nodes + r_nodes + [if_node, ident], "wb_top1_if", list(g0.input),
                     [H.make_tensor_value_info("output0", TP.FLOAT, [1, 300, 6])], initializer=stem_inits + r_inits)
model = H.make_model(graph, opset_imports=list(models[0].opset_import), ir_version=models[0].ir_version)
dst = f"{OUT}/wb_top1_if.onnx"; onnx.save(model, dst)
onnx.checker.check_model(dst)
so = ort.SessionOptions(); so.intra_op_num_threads = 8
o = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"]).run(None, {"images": x.numpy()})[0]
kept = int((ref[0, :, 4] > 0.25).sum())
print(f"wb_top1_if: router picked expert {idx_ref}; ORT vs torch (top-1 eval path) rows conf>0.25: {kept}, max abs diff {np.abs(o[0, :kept] - ref[0, :kept]).max():.3e}; "
      f"{round(len(model.SerializeToString()) / 1e6, 1)} MB; If nodes 3 (nested), branches {E}")
