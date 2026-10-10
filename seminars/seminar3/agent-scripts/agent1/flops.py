"""Agent 1: executed MACs and parameters of YOLO26-M at 640, split by region. CPU, 2 threads."""
import json, torch
torch.set_num_threads(2)
from torch import nn
from ultralytics import YOLO
from ultralytics.nn.modules.moe.weight_bank import bank_modules, BankConv2d
from ultralytics.nn.modules.block import Attention

m = YOLO("yolo26m.yaml").model.float().eval().fuse()
rows = []
def hook(name):
    def f(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            k = mod.kernel_size[0] * mod.kernel_size[1]
            macs = out.numel() * k * mod.in_channels // mod.groups
            p = sum(t.numel() for t in mod.parameters())
            rows.append(dict(name=name, k=mod.kernel_size[0], cin=mod.in_channels, cout=mod.out_channels, g=mod.groups,
                             hw=out.shape[-1], macs=macs, params=p, kind="conv"))
        elif isinstance(mod, nn.Linear):
            rows.append(dict(name=name, k=0, cin=mod.in_features, cout=mod.out_features, g=1, hw=1,
                             macs=out.numel() * mod.in_features, params=sum(t.numel() for t in mod.parameters()), kind="fc"))
        elif isinstance(mod, Attention):
            B, C, Hh, W = inp[0].shape; N = Hh * W
            macs = mod.num_heads * N * N * (mod.key_dim + mod.head_dim)
            rows.append(dict(name=name, k=0, cin=C, cout=C, g=1, hw=Hh, macs=macs, params=0, kind="attn-matmul"))
    return f
for n, mod in m.named_modules():
    if isinstance(mod, (nn.Conv2d, nn.Linear, Attention)):
        mod.register_forward_hook(hook(n))
with torch.no_grad():
    m(torch.zeros(1, 3, 640, 640))
lay = lambda r: int(r["name"].split(".")[1])
tot = sum(r["macs"] for r in rows); totp = sum(r["params"] for r in rows)
allp = sum(p.numel() for p in m.parameters())
print(f"executed MACs {tot/1e9:.3f} G  (GFLOPs=2xMACs {2*tot/1e9:.2f});  params in executed conv/fc {totp/1e6:.3f} M; all params in fused module tree {allp/1e6:.3f} M")
def region(lo, hi, pred=lambda r: True):
    s = [r for r in rows if lo <= lay(r) <= hi and pred(r)]
    return sum(r["macs"] for r in s) / 1e9, sum(r["params"] for r in s) / 1e6, len(s)
for nm, lo, hi in (("stem 0-5", 0, 5), ("layers 6-22", 6, 22), ("detect 23", 23, 23), ("layers 6-23", 6, 23),
                   ("backbone 6-10", 6, 10), ("neck 11-22", 11, 22), ("layers 0-10", 0, 10), ("0-16", 0, 16), ("17-23", 17, 23)):
    a, b, c = region(lo, hi); print(f"{nm:14s} MACs {a:7.3f} G ({100*a*1e9/tot:5.1f}%)  params {b:7.3f} M  n={c}")
one = lambda r: r["kind"] == "conv" and r["k"] == 1 and r["g"] == 1
a, b, c = region(6, 22, one); print(f"1x1 g=1 convs in 6-22: MACs {a:.3f} G ({100*a*1e9/tot:.1f}%) params {b:.3f} M n={c}")
a, b, c = region(6, 22, lambda r: r["kind"] == "conv" and r["k"] == 3 and r["g"] == 1); print(f"3x3 g=1 convs in 6-22: MACs {a:.3f} G params {b:.3f} M n={c}")
a, b, c = region(6, 22, lambda r: r["kind"] == "conv" and r["g"] > 1); print(f"depthwise convs in 6-22: MACs {a:.3f} G params {b:.3f} M n={c}")
a, b, c = region(23, 23, lambda r: True); 
for l in range(24):
    a, b, c = region(l, l); print(f"  layer {l:2d}: MACs {a:6.3f} G params {b:6.3f} M convs {c}")
# routed set as the repository defines it
w = YOLO("yolo26m-wb.yaml").model.float().eval()
bm = bank_modules(w)
names = {id(b): n for n, b in w.named_modules() if isinstance(b, BankConv2d)}
st = bm[0].state
print("bank modules", len(bm), "E", st.num_experts, "k", st.top_k)
per = sum(b.in_channels * b.out_channels for b in bm)
print(f"bank kernel params per expert {per/1e6:.3f} M; held x{st.num_experts} = {per*st.num_experts/1e6:.3f} M")
outc = [(names[id(b)], b.out_channels) for b in bm if names[id(b)].count(".") == 3 and ".cv2." in names[id(b)] or names[id(b)].endswith(".cv2.conv") and names[id(b)].count(".") == 3]
print("block-output banks:", outc, "sum cout", sum(c for _, c in outc))
print("sum cout all 39:", sum(b.out_channels for b in bm), " rank-16 params/expert:", sum(16 * b.out_channels for b in bm))
r = bm[0].router; rp = sum(p.numel() for p in r.parameters()); print("router params", rp, "c1", r.c1)
print("wb model params total", sum(p.numel() for p in w.parameters()) / 1e6, "M (unfused, includes one2many head if present)")
d = YOLO("yolo26m.yaml").model; print("dense unfused params", sum(p.numel() for p in d.parameters()) / 1e6)
json.dump(rows, open("/data/tmp/ds-yolo/seminar3/work/agent1/rows.json", "w"))
