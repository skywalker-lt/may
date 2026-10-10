"""Conditional graph of the merged construct (direction 1's routed shrink + direction 4's attention on the small rung).
Input [1,3,640,640] letterbox. Router: linear Resize to 320 -> YOLO26-N stem (stem_n_320.onnx, pooled 128-d) -> Gemm (ridge
fitted on 12,000 train2017 images to log1p(#boxes under 96 px)) -> score < thr (train-median) -> If.
then (shrink): Resize to 512 -> YOLO26-M@512 + zero-initialised P4 C2PSA block -> boxes x 1.25 (back to 640 coordinates).
else: YOLO26-M@640 as released. Output [1,300,6] in 640-letterbox coordinates in both branches.
With the block zero-initialised, the engine's AP must equal the CPU mixture of the public 512 and 640 dumps under the same route
(parity check), and the then-branch carries the attention block the trained version will use (build check)."""
import copy, numpy as np, onnx, json
from onnx import helper as H, numpy_helper as NH, TensorProto as TP
W = '/data/tmp/ds-yolo/seminar5/work/agent4/'; REQ = W + 'request/'
def renamed(model, pre, keep):
    g = model.graph; nodes = [copy.deepcopy(n) for n in g.node]; ren = lambda s: s if (s in keep or s == '') else pre + s
    for n in nodes: n.name = pre + n.name; n.input[:] = [ren(s) for s in n.input]; n.output[:] = [ren(s) for s in n.output]
    inits = []
    for t in g.initializer: t2 = copy.deepcopy(t); t2.name = pre + t.name; inits.append(t2)
    return nodes, inits, pre + g.output[0].name
stem = onnx.load('/data/tmp/ds-yolo/phase2/onnx/stem_n_320.onnx'); b0 = onnx.load(REQ + 'yolo26m_p4attn0_512.onnx'); b1 = onnx.load(REQ + 'yolo26m_ref640.onnx')
rt = np.load(W + 'r2/router_train.npz'); w, b, thr = rt['w'].astype(np.float32), float(rt['b']), float(rt['thr'])
sn, si, pooled = renamed(stem, 'r_', {'images_r'})
for n in sn: n.input[:] = ['images_r' if s == 'r_images' else s for s in n.input]
inits = si + [NH.from_array(np.array([1, 1, 0.5, 0.5], np.float32), 'sc_r'), NH.from_array(np.array([1, 1, 0.8, 0.8], np.float32), 'sc_b'),
              NH.from_array(w.reshape(1, -1), 'rw'), NH.from_array(np.array([b], np.float32), 'rb'), NH.from_array(np.array(thr, np.float32), 'rthr'),
              NH.from_array(np.zeros(0, np.int64), 'shape0')]
nodes = [H.make_node('Resize', ['images', '', 'sc_r'], ['images_r'], mode='linear', name='router/resize')] + sn + [
    H.make_node('Reshape', [pooled, NH.from_array(np.array([1, -1], np.int64)).name if False else 'pshape'], ['pooled2'], name='router/flat'),
    H.make_node('Gemm', ['pooled2', 'rw', 'rb'], ['r_score'], transB=1, name='router/fc'),
    H.make_node('Reshape', ['r_score', 'shape0'], ['r_s'], name='router/scalar'),
    H.make_node('Less', ['r_s', 'rthr'], ['shrink'], name='router/less')]
inits.append(NH.from_array(np.array([1, -1], np.int64), 'pshape'))
t_n, t_i, t_o = renamed(b0, 'e0_', {'images_b0'})
for n in t_n: n.input[:] = ['images_b0' if s == 'e0_images' else s for s in n.input]
t_i.append(NH.from_array(np.array([1.25, 1.25, 1.25, 1.25, 1, 1], np.float32).reshape(1, 1, 6), 'e0_boxscale'))
t_n = [H.make_node('Resize', ['images', '', 'sc_b'], ['images_b0'], mode='linear', name='e0_resize')] + t_n + [H.make_node('Mul', [t_o, 'e0_boxscale'], ['e0_out640'], name='e0_rescale')]
e_n, e_i, e_o = renamed(b1, 'e1_', {'images'})
for n in e_n: n.input[:] = ['images' if s == 'e1_images' else s for s in n.input]
g0 = H.make_graph(t_n, 'shrink512_p4attn', [], [H.make_tensor_value_info('e0_out640', TP.FLOAT, [1, 300, 6])], initializer=t_i)
g1 = H.make_graph(e_n, 'm640', [], [H.make_tensor_value_info(e_o, TP.FLOAT, [1, 300, 6])], initializer=e_i)
nodes += [H.make_node('If', ['shrink'], ['sel'], then_branch=g0, else_branch=g1, name='If_shrink'), H.make_node('Identity', ['sel'], ['output0'], name='out')]
graph = H.make_graph(nodes, 'if_shrink_attn', [H.make_tensor_value_info('images', TP.FLOAT, [1, 3, 640, 640])], [H.make_tensor_value_info('output0', TP.FLOAT, [1, 300, 6])], initializer=inits)
model = H.make_model(graph, opset_imports=list(b1.opset_import), ir_version=b1.ir_version)
onnx.save(model, REQ + 'if_shrink_p4attn0.onnx'); onnx.checker.check_model(REQ + 'if_shrink_p4attn0.onnx'); print('ok', round(len(model.SerializeToString())/1e6, 1), 'MB')
# router alone (same nodes), for the CPU route list
rnodes = nodes[:-2]; rg = H.make_graph(rnodes, 'router', [H.make_tensor_value_info('images', TP.FLOAT, [1, 3, 640, 640])],
                                      [H.make_tensor_value_info('r_s', TP.FLOAT, []), H.make_tensor_value_info('shrink', TP.BOOL, [])], initializer=inits)
onnx.save(H.make_model(rg, opset_imports=list(b1.opset_import), ir_version=b1.ir_version), W + 'r2/router_only.onnx'); print('router ok')
