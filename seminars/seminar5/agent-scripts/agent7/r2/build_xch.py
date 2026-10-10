"""Exchange engine graphs for an H200 build/profile check (structure, not T4 timing): router = N stem at 320 on a thumbnail
of the input + pool + linear; If -> branch0 = Resize + YOLO26-M@512, branch1 = YOLO26-L@640.
 v1: input 640, branch1 fed DIRECTLY from the engine input (the if_res2 768-branch pattern that cost +0.65 ms on the T4).
 v2: input 768, branch1 fed through Resize 768->640 (every branch behind a Resize).
_f0 / _f1: router bias forces branch 0 / 1 (the Gemm stays data-dependent, so the If is not folded)."""
import copy, numpy as np, onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
from onnx.utils import Extractor
S = "/data/tmp/ds-yolo/seminar5/work/agent7/r2/src"; OUT = "/data/tmp/ds-yolo/seminar5/work/agent7/request"
def out_of(g, prefix):
    last = None
    for n in g.node:
        if n.name.startswith(prefix): last = n.output[0]
    return last
def renamed(model, pre, inp):
    g = model.graph; nodes = [copy.deepcopy(n) for n in g.node]; gin = g.input[0].name
    ren = lambda s: inp if s == gin else ("" if s == "" else pre + s)
    for n in nodes: n.name = pre + n.name; n.input[:] = [ren(s) for s in n.input]; n.output[:] = [ren(s) for s in n.output]
    inits = []
    for t in g.initializer: t2 = copy.deepcopy(t); t2.name = pre + t.name; inits.append(t2)
    return nodes, inits, pre + g.output[0].name
n320 = onnx.shape_inference.infer_shapes(onnx.load(f"{S}/yolo26n_320.onnx")); L5 = out_of(n320.graph, "/model.5/")
stem = Extractor(n320).extract_model([n320.graph.input[0].name], [L5])
m512, l640 = onnx.load(f"{S}/yolo26m_512.onnx"), onnx.load(f"{S}/yolo26l_640.onnx")
for v, size in (("v1", 640), ("v2", 768)):
    for force in ("f0", "f1"):
        sn, si, s5 = renamed(stem, "r_", "images_r")
        rb = np.array([1e4, 0], np.float32) if force == "f0" else np.array([0, 1e4], np.float32)
        inits = si + [NH.from_array(np.array([1, 1, 320 / size, 320 / size], np.float32), "sc_r"), NH.from_array(np.array([1, 1, 512 / size, 512 / size], np.float32), "sc_b0"),
                      NH.from_array(np.array([1, 1, 640 / size, 640 / size], np.float32), "sc_b1"), NH.from_array(np.array([2, 3], np.int64), "r_ax"),
                      NH.from_array((np.random.RandomState(0).randn(2, 128) * 0.1).astype(np.float32), "rw"), NH.from_array(rb, "rb"), NH.from_array(np.array(0, np.int64), "k0")]
        nodes = [H.make_node("Resize", ["images", "", "sc_r"], ["images_r"], mode="linear", name="router/resize")] + sn + [
            H.make_node("ReduceMean", [s5, "r_ax"], ["r_pool"], keepdims=0, name="router/pool"),
            H.make_node("Gemm", ["r_pool", "rw", "rb"], ["r_logits"], transB=1, name="router/fc"), H.make_node("ArgMax", ["r_logits"], ["r_idx"], axis=1, keepdims=0, name="router/argmax"),
            H.make_node("Squeeze", ["r_idx"], ["r_i"], name="router/sq"), H.make_node("Equal", ["r_i", "k0"], ["is0"], name="router/is0")]
        b0n, b0i, b0o = renamed(m512, "e0_", "images_b0"); b0n = [H.make_node("Resize", ["images", "", "sc_b0"], ["images_b0"], mode="linear", name="e0_resize")] + b0n
        if v == "v1": b1n, b1i, b1o = renamed(l640, "e1_", "images")
        else:
            b1n, b1i, b1o = renamed(l640, "e1_", "images_b1"); b1n = [H.make_node("Resize", ["images", "", "sc_b1"], ["images_b1"], mode="linear", name="e1_resize")] + b1n
        g0 = H.make_graph(b0n, "m512", [], [H.make_tensor_value_info(b0o, TP.FLOAT, [1, 300, 6])], initializer=b0i)
        g1 = H.make_graph(b1n, "l640", [], [H.make_tensor_value_info(b1o, TP.FLOAT, [1, 300, 6])], initializer=b1i)
        nodes += [H.make_node("If", ["is0"], ["sel"], then_branch=g0, else_branch=g1, name="If_xch"), H.make_node("Identity", ["sel"], ["output0"], name="out")]
        graph = H.make_graph(nodes, f"xch_{v}", [H.make_tensor_value_info("images", TP.FLOAT, [1, 3, size, size])], [H.make_tensor_value_info("output0", TP.FLOAT, [1, 300, 6])], initializer=inits)
        model = H.make_model(graph, opset_imports=list(l640.opset_import), ir_version=l640.ir_version)
        p = f"{OUT}/xch_{v}_{force}.onnx"; onnx.save(model, p); onnx.checker.check_model(p); print(p, "ok", round(len(model.SerializeToString()) / 1e6, 1), "MB", flush=True)
