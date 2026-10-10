"""Agent 4: per-layer parameters and executed MACs of YOLO26-M at 640 (fused, deploy graph), and option accounting."""
import collections, json, torch
from torch import nn
torch.set_num_threads(2)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.moe.weight_bank import is_bankable
from ultralytics.nn.modules.conv import Conv

m = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False).eval()
m.fuse(verbose=False) if hasattr(m, "fuse") else None
macs = collections.Counter(); elems = {}
def hook(name, top):
    def f(mod, i, o):
        if isinstance(mod, nn.Conv2d):
            k = mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
            macs[name] += o.numel() * k; elems[name] = o.numel()
        elif isinstance(mod, nn.Linear):
            macs[name] += o.numel() * mod.in_features
    return f
att = collections.Counter()
for name, mod in m.named_modules():
    if isinstance(mod, (nn.Conv2d, nn.Linear)):
        mod.register_forward_hook(hook(name, None))
with torch.no_grad():
    m(torch.zeros(1, 3, 640, 640))
top = lambda n: int(n.split(".")[1])
L = len(m.model)
pl = collections.Counter(); ml = collections.Counter()
for n, p in m.named_parameters(): pl[top(n)] += p.numel()
for n, v in macs.items(): ml[top(n)] += v
print("layer type params MACs(G)")
pass
P = sum(pl.values()); M = sum(ml.values())
print("TOTAL params", P, "MACs G", M / 1e9, "GFLOPs(2xMAC)", 2 * M / 1e9)
seg = lambda a, b: (sum(pl[i] for i in range(a, b + 1)), sum(ml[i] for i in range(a, b + 1)))
for a, b in [(0, 5), (6, 22), (23, 23), (6, 23), (6, 10), (11, 22), (11, 23), (7, 23), (9, 23)]:
    p, mm = seg(a, b); print(f"layers {a}-{b}: params {p} ({p/P:.3f}) MACs {mm/1e9:.3f}G ({mm/M:.3f})")
# bankable 1x1 convs in 6-22
bank = []
for name, mod in m.named_modules():
    if isinstance(mod, Conv) and 6 <= top(name) <= 22 and mod.conv.kernel_size == (1, 1) and mod.conv.groups == 1 and mod.conv.stride == (1, 1):
        c = mod.conv; bank.append((name, c.in_channels, c.out_channels, c.weight.numel() + (c.bias.numel() if c.bias is not None else 0), macs[name + ".conv"], elems[name + ".conv"]))
nb = len(bank); bp = sum(b[3] for b in bank); bm = sum(b[4] for b in bank); be = sum(b[5] for b in bank)
print("bankable 1x1 in 6-22:", nb, "params", bp, "MACs G", bm / 1e9, "share of MACs", bm / M, "out elems", be)
sc2 = sum(b[2] for b in bank); sc1 = sum(b[1] for b in bank)
print("sum cout", sc2, "sum cin", sc1, "rank16 B params/expert", 16 * sc2, "shared A", 16 * sc1, "rank16 extra MACs G", sum(16 * b[1] * b[5] / b[2] + 16 * b[5] for b in bank) / 1e9)
print("per-layer scale/shift MACs (elems)", be / 1e9)
outs = [b for b in bank if b[0].endswith("cv2") and b[0].count(".") == 2]
print("block-output convs:", [(b[0], b[2]) for b in outs], "sum cout", sum(b[2] for b in outs), "elems", sum(b[5] for b in outs))
print("attention/PSA non-conv matmul not counted; bank per top layer:", {i: (sum(1 for b in bank if top(b[0]) == i), sum(b[3] for b in bank if top(b[0]) == i)) for i in range(6, 23) if any(top(b[0]) == i for b in bank)})
print("3x3+other params in 6-22:", sum(pl[i] for i in range(6, 23)) - bp)

