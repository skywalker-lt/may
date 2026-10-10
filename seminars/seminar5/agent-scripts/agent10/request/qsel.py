# Which Conv nodes of the YOLO26-M ONNX get INT8 Q/DQ, and which activation tensors feed them.
import re, onnx
def layer(name):
    m = re.match(r"/model\.(\d+)/", name); return int(m.group(1)) if m else -1
def keep_fp(name):
    """kept in floating point: the first conv (image input), the PSA attention blocks (layer 10 m, layer 22 m.0.1),
    and the head's final linear convs (box / class outputs)."""
    if layer(name) == 0: return True
    if name.startswith("/model.10/m/") or name.startswith("/model.22/m.0/m.0.1/"): return True
    if re.search(r"one2one_cv[23]\.\d/one2one_cv[23]\.\d\.2/Conv$", name): return True
    return False
def quant_convs(g, scope="all"):
    out = []
    for n in g.node:
        if n.op_type != "Conv" or keep_fp(n.name): continue
        if scope == "stem" and not (1 <= layer(n.name) <= 5): continue
        if scope == "tail" and not (layer(n.name) >= 6): continue
        if scope.startswith("L"):
            lo, hi = map(int, scope[1:].split("-"))
            if not (lo <= layer(n.name) <= hi): continue
        out.append(n)
    return out
