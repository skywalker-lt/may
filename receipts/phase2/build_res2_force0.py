"""Two-scale engine: input 768x768; router = N stem at 320 (Resize) + pooling + linear; If -> branch0 = Resize to 512 + M@512,
branch1 = M@768. Plus a plain spliced M-stem + L-tail graph (no If). Timing graphs; router weights random."""
import copy, numpy as np, onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
OUT = "/data/tmp/ds-yolo/phase2/onnx"
def out_of(g, prefix):
    last = None
    for n in g.node:
        if n.name.startswith(prefix): last = n.output[0]
    return last
def renamed(model, pre, keep):
    g = model.graph; nodes = [copy.deepcopy(n) for n in g.node]; ren = lambda s: s if (s in keep or s == "") else pre + s
    for n in nodes: n.name = pre + n.name; n.input[:] = [ren(s) for s in n.input]; n.output[:] = [ren(s) for s in n.output]
    inits = []
    for t in g.initializer: t2 = copy.deepcopy(t); t2.name = pre + t.name; inits.append(t2)
    return nodes, inits, pre + g.output[0].name
m512, m768, stem = onnx.load(f"{OUT}/yolo26m_512.onnx"), onnx.load(f"{OUT}/yolo26m_768.onnx"), onnx.load(f"{OUT}/stem_n_320.onnx")
# router on a 320 thumbnail of the 768 input
sn, si, pooled = renamed(stem, "r_", {"images_r"})
for n in sn: n.input[:] = ["images_r" if s == "r_images" else s for s in n.input]
C = pooled  # [1,128] after ReduceMean (keepdims=0)
inits = si + [NH.from_array(np.array([1, 1, 320 / 768, 320 / 768], dtype=np.float32), "sc_r"), NH.from_array(np.array([1, 1, 512 / 768, 512 / 768], dtype=np.float32), "sc_b"),
              NH.from_array((np.random.randn(2, 128) * 0.1).astype(np.float32), "rw"), NH.from_array(np.array([100.0, 0.0], np.float32), "rb"), NH.from_array(np.array(0, dtype=np.int64), "k0")]
nodes = [H.make_node("Resize", ["images", "", "sc_r"], ["images_r"], mode="linear", name="router/resize")] + sn + [
    H.make_node("Gemm", [C, "rw", "rb"], ["r_logits"], transB=1, name="router/fc"), H.make_node("ArgMax", ["r_logits"], ["r_idx"], axis=1, keepdims=0, name="router/argmax"),
    H.make_node("Squeeze", ["r_idx"], ["r_i"], name="router/sq"), H.make_node("Equal", ["r_i", "k0"], ["is0"], name="router/is0")]
b0n, b0i, b0o = renamed(m512, "e0_", {"images_b0"})
for n in b0n: n.input[:] = ["images_b0" if s == "e0_images" else s for s in n.input]
b0n = [H.make_node("Resize", ["images", "", "sc_b"], ["images_b0"], mode="linear", name="e0_resize")] + b0n
b1n, b1i, b1o = renamed(m768, "e1_", {"images"})
for n in b1n: n.input[:] = ["images" if s == "e1_images" else s for s in n.input]
g0 = H.make_graph(b0n, "b512", [], [H.make_tensor_value_info(b0o, TP.FLOAT, [1, 300, 6])], initializer=b0i)
g1 = H.make_graph(b1n, "b768", [], [H.make_tensor_value_info(b1o, TP.FLOAT, [1, 300, 6])], initializer=b1i)
nodes.append(H.make_node("If", ["is0"], ["sel"], then_branch=g0, else_branch=g1, name="If_res")); nodes.append(H.make_node("Identity", ["sel"], ["output0"], name="out"))
graph = H.make_graph(nodes, "if_res2", [H.make_tensor_value_info("images", TP.FLOAT, [1, 3, 768, 768])], [H.make_tensor_value_info("output0", TP.FLOAT, [1, 300, 6])], initializer=inits)
model = H.make_model(graph, opset_imports=list(m768.opset_import), ir_version=m768.ir_version); onnx.save(model, f"{OUT}/if_res2_force0.onnx"); onnx.checker.check_model(f"{OUT}/if_res2_force0.onnx"); print("if_res2 ok", round(len(model.SerializeToString()) / 1e6, 1), "MB")
