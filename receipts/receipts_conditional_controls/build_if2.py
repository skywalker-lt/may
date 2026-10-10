"""Conditional-engine controls requested in seminar 3, round 1 (local tool).
  if_dense4 / if_dense2 : dense YOLO26-M wrapped in the same nested If, identical branches (is 0.966x a compile-path effect?)
  if_distinct4          : four branches in which EVERY conv weight of layers 6-23 differs (1% perturbation per branch)
  router_only           : stem + router -> logits, to compare fp16 and fp32 routing decisions on the T4
  in_branch{i}.npy      : one real val2017 input per branch (largest fp32 margin), for per-branch timing
The router and its weights are those of the upcycled COCO checkpoint used in Phase 1."""
import copy, glob, shutil, sys
import cv2, numpy as np, onnx, onnxruntime as ort, torch
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
from torch import nn
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.moe.weight_bank import BankConv2d, bank_modules

OUT = "/data/tmp/ds-yolo/phase1b/onnx"; WB = "/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt"
lb = LetterBox((640, 640), auto=False)
def load(f): return torch.from_numpy(lb(image=cv2.imread(f))[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]

wb = YOLO(WB).model.float().eval()
st = bank_modules(wb)[0].state; st.top_k = 1; owner = bank_modules(wb)[0]; router = owner.router; E = st.num_experts

def dense_from(i, mode):
    """mode 'expert': kernels of expert i (Phase 1 engine); 'dense': the mean kernel (= YOLO26-M) in every branch;
    'distinct': mean kernel, then every conv weight of layers 6-23 perturbed by 1% with a branch-specific seed."""
    m = copy.deepcopy(wb)
    for blk in m.modules():
        if isinstance(blk, Conv) and isinstance(blk.conv, BankConv2d):
            b = blk.conv; c = nn.Conv2d(b.in_channels, b.out_channels, 1, bias=False)
            w = b.experts[i].data if mode == "expert" else torch.stack(tuple(b.experts)).mean(0)
            c.weight.data.copy_(w); blk.conv = c
    if mode == "distinct":
        g = torch.Generator().manual_seed(1000 + i)
        for layer in list(m.model)[6:24]:
            for mod in layer.modules():
                if isinstance(mod, nn.Conv2d):
                    mod.weight.data.add_(0.01 * mod.weight.data.std() * torch.randn(mod.weight.shape, generator=g))
    m.yaml.pop("weight_bank", None)
    return m

def export(m, name):
    y = YOLO(WB); y.model = m
    f = y.export(format="onnx", imgsz=640, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
    p = f"{OUT}/_{name}.onnx"; shutil.move(f, p); return onnx.load(p)

def out_of(g, prefix):
    last = None
    for n in g.node:
        if n.name.startswith(prefix): last = n.output[0]
    return last

def router_nodes(L5, n_eq):
    n = router.norm; c = lambda name, a: NH.from_array(np.asarray(a), name)
    inits = [c("r_mean", n.running_mean.numpy()[None]), c("r_rstd", torch.rsqrt(n.running_var + n.eps).numpy()[None]),
             c("r_w1", router.fc1.weight.detach().numpy()), c("r_b1", router.fc1.bias.detach().numpy()),
             c("r_w2", router.fc2.weight.detach().numpy()), c("r_b2", router.fc2.bias.detach().numpy()), c("r_axes", np.array([2, 3], dtype=np.int64))]
    nodes = [H.make_node("ReduceMean", [L5, "r_axes"], ["r_pool"], keepdims=0, name="router/pool"),
             H.make_node("Sub", ["r_pool", "r_mean"], ["r_c"], name="router/sub"), H.make_node("Mul", ["r_c", "r_rstd"], ["r_n"], name="router/mul"),
             H.make_node("Gemm", ["r_n", "r_w1", "r_b1"], ["r_h"], transB=1, name="router/fc1"),
             H.make_node("Sigmoid", ["r_h"], ["r_s"], name="router/sig"), H.make_node("Mul", ["r_h", "r_s"], ["r_a"], name="router/silu"),
             H.make_node("Gemm", ["r_a", "r_w2", "r_b2"], ["r_logits"], transB=1, name="router/fc2")]
    if n_eq:
        nodes += [H.make_node("ArgMax", ["r_logits"], ["r_idx"], axis=1, keepdims=0, name="router/argmax"), H.make_node("Squeeze", ["r_idx"], ["r_i"], name="router/squeeze")]
        for i in range(n_eq):
            inits.append(c(f"r_k{i}", np.array(i, dtype=np.int64))); nodes.append(H.make_node("Equal", ["r_i", f"r_k{i}"], [f"r_is{i}"], name=f"router/is{i}"))
    return nodes, inits

def stitch(models, name):
    nb = len(models); g0 = models[0].graph
    L4, L5 = out_of(g0, "/model.4/"), out_of(g0, "/model.5/")
    is_stem = lambda n: any(n.name.startswith(f"/model.{k}/") for k in range(6))
    stem_nodes = [n for n in g0.node if is_stem(n)]
    init0 = {t.name: t for t in g0.initializer}
    stem_inits = [init0[x] for x in {i for n in stem_nodes for i in n.input} if x in init0]
    def branch(i):
        g = models[i].graph; pre = f"e{i}_"; keep = {L4, L5}
        nodes = [copy.deepcopy(n) for n in g.node if not is_stem(n)]
        used = {inp for n in nodes for inp in n.input} & {t.name for t in g.initializer}
        ren = lambda s: s if (s in keep or s == "") else pre + s
        for n in nodes:
            n.name = pre + n.name; n.input[:] = [ren(s) for s in n.input]; n.output[:] = [ren(s) for s in n.output]
        inits = []
        for t in g.initializer:
            if t.name in used:
                t2 = copy.deepcopy(t); t2.name = pre + t.name; inits.append(t2)
        return H.make_graph(nodes, f"expert{i}", [], [H.make_tensor_value_info(pre + g.output[0].name, TP.FLOAT, [1, 300, 6])], initializer=inits)
    def nest(i):
        then_g = branch(i)
        if i == nb - 2: else_g = branch(nb - 1)
        else:
            inner, inner_out = nest(i + 1)
            else_g = H.make_graph([inner], f"else{i}", [], [H.make_tensor_value_info(inner_out, TP.FLOAT, [1, 300, 6])])
        return H.make_node("If", [f"r_is{i}"], [f"sel{i}"], then_branch=then_g, else_branch=else_g, name=f"If_{i}"), f"sel{i}"
    r_nodes, r_inits = router_nodes(L5, nb - 1)
    if_node, sel = nest(0)
    graph = H.make_graph(stem_nodes + r_nodes + [if_node, H.make_node("Identity", [sel], ["output0"], name="out")], name, list(g0.input),
                         [H.make_tensor_value_info("output0", TP.FLOAT, [1, 300, 6])], initializer=stem_inits + r_inits)
    model = H.make_model(graph, opset_imports=list(models[0].opset_import), ir_version=models[0].ir_version)
    dst = f"{OUT}/{name}.onnx"; onnx.save(model, dst); onnx.checker.check_model(dst)
    print(f"{name}: {nb} branches, {round(len(model.SerializeToString()) / 1e6, 1)} MB", flush=True)
    return dst, stem_nodes, stem_inits, L5

dense = export(dense_from(0, "dense"), "dense")
stitch([dense] * 4, "if_dense4"); stitch([dense] * 2, "if_dense2")
dst, stem_nodes, stem_inits, L5 = stitch([export(dense_from(i, "distinct"), f"distinct{i}") for i in range(4)], "if_distinct4")
# router-only graph: stem + router -> logits
r_nodes, r_inits = router_nodes(L5, 0)
g = H.make_graph(stem_nodes + r_nodes, "router_only", list(dense.graph.input), [H.make_tensor_value_info("r_logits", TP.FLOAT, [1, E])], initializer=stem_inits + r_inits)
ro = H.make_model(g, opset_imports=list(dense.opset_import), ir_version=dense.ir_version); onnx.save(ro, f"{OUT}/router_only.onnx"); onnx.checker.check_model(f"{OUT}/router_only.onnx")
# one real input per branch (largest fp32 margin among 400 val images), checked against the router-only graph in ORT
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(f"{OUT}/router_only.onnx", so, providers=["CPUExecutionProvider"])
files = sorted(glob.glob("/data/datasets/coco/images/val2017/*.jpg"))[::12][:400]; best = {}
for f in files:
    x = load(f).numpy(); lg = sess.run(None, {"images": x})[0][0]; o = np.sort(lg); i = int(lg.argmax()); mg = float(o[-1] - o[-2])
    if i not in best or mg > best[i][0]: best[i] = (mg, f, x)
for i in range(E):
    mg, f, x = best[i]; np.save(f"{OUT}/in_branch{i}.npy", x.astype(np.float32)); print(f"in_branch{i}: {f.split('/')[-1]} margin {mg:.3f}")
