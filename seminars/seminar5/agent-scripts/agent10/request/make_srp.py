# Saturation-routed precision (SRP) as ONE ONNX graph: INT8 stem (layers 0-5, Q/DQ on 1-5), a saturation tap on the
# layer-5 output (max|x|/amax5, fraction |x|>amax5, pooled count head), a 3-input linear score, and one If:
#   then (score > tau): the fp16 tail  (layers 6-23 without Q/DQ)
#   else              : the INT8 tail  (layers 6-23 with Q/DQ)
# The taps are also graph outputs so a dump can record them per image. Inputs: the two Q/DQ ONNX files from
# make_qdq.py (scope all and scope stem: their stems are identical). Router weights are arguments (fit on train2017 later).
import os, sys, json, copy, numpy as np, onnx
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
def stem_names(g):
    import re
    def lay(n):
        m = re.match(r"/model\.(\d+)/", n.name)
        if m: return int(m.group(1))
        for o in n.input:                      # Q/DQ nodes are named after their tensor
            m = re.match(r"/model\.(\d+)/", o)
            if m and n.op_type in ("QuantizeLinear", "DequantizeLinear"): return int(m.group(1))
        return -1
    return lay
def build(all_q, stem_q, amax_json, dst, w=(1.0, 1.0, 0.0), b=0.0, tau=1.5, count_w=None, method="entropy"):
    A = json.load(open(amax_json)); MA, MS = onnx.load(all_q), onnx.load(stem_q)
    ga, gs = MA.graph, MS.graph; lay = stem_names(ga)
    # a node belongs to the stem if it is a layer 0-5 node or a Q/DQ feeding a layer 1-5 conv (by its consumer)
    cons = {}
    for n in ga.node:
        for i in n.input: cons.setdefault(i, []).append(n)
    def in_stem(n, g_cons):
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            outs = [c for o in n.output for c in g_cons.get(o, [])]
            while outs and outs[0].op_type == "DequantizeLinear": outs = [c for o in outs[0].output for c in g_cons.get(o, [])]
            return bool(outs) and all(0 <= lay(c) <= 5 for c in outs)
        return 0 <= lay(n) <= 5
    cons_s = {}
    for n in gs.node:
        for i in n.input: cons_s.setdefault(i, []).append(n)
    stem_a = [n for n in ga.node if in_stem(n, cons)]
    stem_s = {n.name for n in gs.node if in_stem(n, cons_s)}
    assert {n.name for n in stem_a} == stem_s, "stems differ"
    outer = {"images"} | {o for n in stem_a for o in n.output}
    init_a = {t.name: t for t in ga.initializer}; init_s = {t.name: t for t in gs.initializer}
    outer_inits = [init_a[i] for i in sorted({i for n in stem_a for i in n.input if i in init_a})]
    L5 = [n.output[0] for n in ga.node if n.name == "/model.5/act/Mul"][0]
    def branch(g, inits, pre, name):
        nodes = [copy.deepcopy(n) for n in g.node if n.name not in stem_s]
        used = sorted({i for n in nodes for i in n.input if i in inits})
        ren = lambda s: s if (s in outer or s == "") else pre + s
        for n in nodes:
            n.name = pre + n.name; n.input[:] = [ren(s) for s in n.input]; n.output[:] = [ren(s) for s in n.output]
        its = []
        for u in used:
            t = copy.deepcopy(inits[u]); t.name = pre + u; its.append(t)
        return H.make_graph(nodes, name, [], [H.make_tensor_value_info(pre + g.output[0].name, TP.FLOAT, [1, 300, 6])], initializer=its)
    then_g = branch(gs, init_s, "fp_", "tail_fp16")      # stem-only quantised model: its tail is plain floating point
    else_g = branch(ga, init_a, "q8_", "tail_int8")      # all-quantised model: its tail carries Q/DQ
    a5 = A[L5][method]; c = lambda n, v: NH.from_array(np.asarray(v, np.float32), n)
    C5 = 512 if count_w is None else len(count_w)
    r_inits = [c("srp_amax5", a5), c("srp_w0", w[0]), c("srp_w1", w[1]), c("srp_w2", w[2]), c("srp_b", b), c("srp_tau", tau),
               c("srp_cw", np.zeros((C5, 1), np.float32) if count_w is None else np.asarray(count_w, np.float32).reshape(-1, 1)),
               NH.from_array(np.array([2, 3], np.int64), "srp_ax23")]
    r_nodes = [H.make_node("Abs", [L5], ["srp_abs"]),
               H.make_node("ReduceMax", ["srp_abs"], ["srp_max"], keepdims=0),
               H.make_node("Div", ["srp_max", "srp_amax5"], ["srp_maxr"]),
               H.make_node("Greater", ["srp_abs", "srp_amax5"], ["srp_gt"]),
               H.make_node("Cast", ["srp_gt"], ["srp_gtf"], to=TP.FLOAT),
               H.make_node("ReduceMean", ["srp_gtf"], ["srp_frac"], keepdims=0),
               H.make_node("ReduceMean", [L5, "srp_ax23"], ["srp_pool"], keepdims=0),      # [1, C]
               H.make_node("MatMul", ["srp_pool", "srp_cw"], ["srp_cnt2"]),
               H.make_node("ReduceSum", ["srp_cnt2"], ["srp_cnt"], keepdims=0),
               H.make_node("Mul", ["srp_maxr", "srp_w0"], ["srp_t0"]), H.make_node("Mul", ["srp_frac", "srp_w1"], ["srp_t1"]),
               H.make_node("Mul", ["srp_cnt", "srp_w2"], ["srp_t2"]),
               H.make_node("Sum", ["srp_t0", "srp_t1", "srp_t2", "srp_b"], ["srp_score"]),
               H.make_node("Greater", ["srp_score", "srp_tau"], ["srp_cond"]),
               H.make_node("If", ["srp_cond"], ["output0"], then_branch=then_g, else_branch=else_g, name="srp_if")]
    G = H.make_graph(list(stem_a) + r_nodes, "srp", [copy.deepcopy(ga.input[0])],
                     [H.make_tensor_value_info("output0", TP.FLOAT, [1, 300, 6]),
                      H.make_tensor_value_info("srp_maxr", TP.FLOAT, []), H.make_tensor_value_info("srp_frac", TP.FLOAT, []),
                      H.make_tensor_value_info("srp_score", TP.FLOAT, [])], initializer=outer_inits + r_inits)
    M = H.make_model(G, opset_imports=MA.opset_import, ir_version=MA.ir_version); onnx.checker.check_model(M); onnx.save(M, dst)
    print(dst, "outer nodes", len(stem_a) + len(r_nodes), "then", len(then_g.node), "else", len(else_g.node), "amax5 %.3f" % a5)
if __name__ == "__main__":
    a = sys.argv; build(a[1], a[2], a[3], a[4], tau=float(a[5]) if len(a) > 5 else 1.5, method=a[6] if len(a) > 6 else "entropy")
