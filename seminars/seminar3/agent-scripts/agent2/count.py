"""Agent 2: MACs/params of deployed YOLO26-M and the routed variants (CPU, 2 threads)."""
import re, collections, torch, torch.nn as nn
torch.set_num_threads(2)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

def build(cfg):
    m = DetectionModel(cfg, ch=3, nc=80, verbose=False).eval()
    return m

m = build("yolo26m.yaml")
print("unfused params", sum(p.numel() for p in m.parameters()), "get_flops", get_flops(m, 640))
m = m.fuse(verbose=False)
print("fused params", sum(p.numel() for p in m.parameters()), "get_flops(fused)", get_flops(m, 640))
recs = []
def hook(name):
    def f(mod, i, o):
        k = mod.kernel_size[0] * mod.kernel_size[1]
        macs = o.numel() * k * mod.in_channels // mod.groups
        recs.append((name, mod.kernel_size[0], mod.in_channels, mod.out_channels, mod.groups, o.shape[-1], macs,
                     sum(p.numel() for p in mod.parameters()), o.numel()))
    return f
for n, mod in m.named_modules():
    if isinstance(mod, nn.Conv2d):
        mod.register_forward_hook(hook(n))
with torch.no_grad():
    m(torch.zeros(1, 3, 640, 640))
def idx(n): return int(re.match(r"model\.(\d+)", n).group(1))
tot_m = sum(r[6] for r in recs); tot_p = sum(r[7] for r in recs)
print(f"executed convs {len(recs)}  MACs {tot_m/1e9:.3f} G  (GFLOPs x2 = {2*tot_m/1e9:.2f})  conv params executed {tot_p/1e6:.3f} M")
reg = collections.OrderedDict()
for lo, hi, lab in [(0, 5, "stem 0-5"), (6, 22, "body 6-22"), (23, 23, "head 23")]:
    rr = [r for r in recs if lo <= idx(r[0]) <= hi]
    reg[lab] = rr
    print(f"{lab:10} convs {len(rr):3d} MACs {sum(r[6] for r in rr)/1e9:.3f} G  params {sum(r[7] for r in rr)/1e6:.3f} M")
body = reg["body 6-22"]
one = [r for r in body if r[1] == 1 and r[4] == 1]
print(f"1x1 g=1 convs in 6-22: {len(one)}  params {sum(r[7] for r in one)/1e6:.3f} M  MACs {sum(r[6] for r in one)/1e9:.3f} G  out elems {sum(r[8] for r in one)/1e6:.2f} M  out channels {sum(r[3] for r in one)}")
k3 = [r for r in body if r[1] == 3]
print(f"3x3 convs in 6-22: {len(k3)} params {sum(r[7] for r in k3)/1e6:.3f} M MACs {sum(r[6] for r in k3)/1e9:.3f} G")
oth = [r for r in body if not (r[1] == 1 and r[4] == 1) and r[1] != 3]
print("other convs in 6-22:", [(r[0], r[1], r[4], r[7]) for r in oth])
# block outputs: the cv2 (C3k2 output) convs and plain block outputs
for r in one: print("  ", r[0], r[2], "->", r[3], "hw", r[5], "p", r[7], "macs %.1fM" % (r[6]/1e6))
# rank-16 delta params per expert per layer: 16*(cin+cout)
print("rank-16 params/expert, 39 layers:", sum(16 * (r[2] + r[3]) for r in one))
# attention matmuls are not conv-hooked; non-conv params:
print("non-conv params:", sum(p.numel() for p in m.parameters()) - tot_p)
# routed model
try:
    w = build("yolo26m-wb.yaml")
    print("wb unfused params", sum(p.numel() for p in w.parameters()))
    from ultralytics.nn.modules.moe.weight_bank import bank_modules, WeightBankRouter
    bm = bank_modules(w)
    print("bank modules", len(bm), "bank params", sum(p.numel() for b in bm for p in b.parameters() if p.ndim >= 2 or True))
    rt = [x for x in w.modules() if isinstance(x, WeightBankRouter)]
    print("routers", len(rt), "router params", sum(p.numel() for x in rt for p in x.parameters()))
except Exception as e:
    print("wb build failed:", repr(e))
