"""RS-2 engine graph (routed shrink, two rungs, one weight set), 640 input, batch 1, static.
router: Resize 640->320 (linear) + YOLO26-N layers 0-5 + global pool + standardise + linear score (train2017 ridge, target
log1p(small+medium count)); cond = score < threshold (train2017 quantile at 512 share 0.5) -> then: Resize 640->512 + M@512,
boxes x1.25 back to the 640 frame; else: M@640. Variant 'fed' feeds the else branch through a scale-1 Resize (agent 7's
countermeasure to the +0.65 ms directly-fed-branch penalty seen in if_res2); variant 'direct' feeds it the engine input."""
import copy, sys, numpy as np, onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
SRC = "/data/tmp/ds-yolo/phase2/onnx"; OUT = "/data/tmp/ds-yolo/seminar5/work/agent1/request"
R = np.load("/data/tmp/ds-yolo/seminar5/work/agent1/r2/router_train.npz")
def renamed(model, pre, keep):
    g = model.graph; nodes = [copy.deepcopy(n) for n in g.node]; ren = lambda s: s if (s in keep or s == "") else pre + s
    for n in nodes: n.name = pre + n.name; n.input[:] = [ren(s) for s in n.input]; n.output[:] = [ren(s) for s in n.output]
    inits = []
    for t in g.initializer: t2 = copy.deepcopy(t); t2.name = pre + t.name; inits.append(t2)
    return nodes, inits, pre + g.output[0].name
def build(variant):
    m512, m640, stem = onnx.load(f"{SRC}/yolo26m_512.onnx"), onnx.load(f"{SRC}/_m640.onnx"), onnx.load(f"{SRC}/stem_n_320.onnx")
    sn, si, pooled = renamed(stem, "r_", {"images_r"})
    for n in sn: n.input[:] = ["images_r" if s == "r_images" else s for s in n.input]
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    inits = si + [NH.from_array(f32([1, 1, 0.5, 0.5]), "sc_r"), NH.from_array(f32(R["mu"])[None], "r_mu"), NH.from_array(f32(1.0 / R["sd"])[None], "r_rsd"),
                  NH.from_array(f32(R["w"]).reshape(1, -1), "r_w"), NH.from_array(f32([R["b"]]), "r_b"), NH.from_array(f32(R["thr"]).reshape(()), "r_thr")]
    nodes = [H.make_node("Resize", ["images", "", "sc_r"], ["images_r"], mode="linear", name="router/resize")] + sn + [
        H.make_node("Sub", [pooled, "r_mu"], ["r_c"], name="router/sub"), H.make_node("Mul", ["r_c", "r_rsd"], ["r_z"], name="router/mul"),
        H.make_node("Gemm", ["r_z", "r_w", "r_b"], ["r_s"], transB=1, name="router/fc"), H.make_node("Squeeze", ["r_s"], ["r_s0"], name="router/sq"),
        H.make_node("Less", ["r_s0", "r_thr"], ["shrink"], name="router/less")]
    b0n, b0i, b0o = renamed(m512, "e0_", {"images_b0"})
    for n in b0n: n.input[:] = ["images_b0" if s == "e0_images" else s for s in n.input]
    b0i = b0i + [NH.from_array(f32([1, 1, 0.8, 0.8]), "sc_b"), NH.from_array(f32([1.25, 1.25, 1.25, 1.25, 1, 1]).reshape(1, 1, 6), "box_up")]
    b0n = [H.make_node("Resize", ["images", "", "sc_b"], ["images_b0"], mode="linear", name="e0_resize")] + b0n + [H.make_node("Mul", [b0o, "box_up"], ["e0_out"], name="e0_boxup")]
    feed = "images_b1" if variant == "fed" else "images"
    b1n, b1i, b1o = renamed(m640, "e1_", {feed})
    for n in b1n: n.input[:] = [feed if s == "e1_images" else s for s in n.input]
    if variant == "fed":
        b1i = b1i + [NH.from_array(f32([1, 1, 1, 1]), "sc_1")]; b1n = [H.make_node("Resize", ["images", "", "sc_1"], ["images_b1"], mode="linear", name="e1_resize")] + b1n
    g0 = H.make_graph(b0n, "b512", [], [H.make_tensor_value_info("e0_out", TP.FLOAT, [1, 300, 6])], initializer=b0i)
    g1 = H.make_graph(b1n, "b640", [], [H.make_tensor_value_info(b1o, TP.FLOAT, [1, 300, 6])], initializer=b1i)
    nodes += [H.make_node("If", ["shrink"], ["sel"], then_branch=g0, else_branch=g1, name="If_rs2"), H.make_node("Identity", ["sel"], ["output0"], name="out")]
    graph = H.make_graph(nodes, f"rs2_{variant}", [H.make_tensor_value_info("images", TP.FLOAT, [1, 3, 640, 640])], [H.make_tensor_value_info("output0", TP.FLOAT, [1, 300, 6])], initializer=inits)
    model = H.make_model(graph, opset_imports=list(m640.opset_import), ir_version=m640.ir_version)
    p = f"{OUT}/rs2_{variant}.onnx"; onnx.save(model, p); onnx.checker.check_model(p); print(p, round(len(model.SerializeToString()) / 1e6, 1), "MB")
    # router-only probe graph (same router, outputs the score) for CPU parity checks
    if variant == "fed":
        rg = H.make_graph(nodes[:-2], "rs2_router", [H.make_tensor_value_info("images", TP.FLOAT, [1, 3, 640, 640])], [H.make_tensor_value_info("r_s0", TP.FLOAT, None)], initializer=inits)
        rm = H.make_model(rg, opset_imports=list(m640.opset_import), ir_version=m640.ir_version); onnx.save(rm, f"/data/tmp/ds-yolo/seminar5/work/agent1/r2/rs2_router.onnx")
for v in ("fed", "direct"): build(v)
