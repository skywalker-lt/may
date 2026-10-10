"""Agent 3: MACs and parameters of YOLO26-M per top-level layer, routed subsets, and option totals (CPU, 2 threads)."""
import torch, torch.nn as nn, json, sys
torch.set_num_threads(2)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops

def build(cfg):
    m = DetectionModel(cfg, ch=3, nc=80, verbose=False)
    m.eval()
    return m

def profile(m, tag):
    recs = []  # (top_idx, name, k, cin, cout, groups, params, macs)
    hooks = []
    def mk(name, mod):
        def h(mod_, inp, out):
            k = mod_.kernel_size[0] * mod_.kernel_size[1]
            macs = out.numel() / out.shape[0] * k * mod_.in_channels / mod_.groups
            p = sum(x.numel() for x in mod_.parameters())
            recs.append((int(name.split('.')[1]), name, mod_.kernel_size[0], mod_.in_channels, mod_.out_channels, mod_.groups, p, macs, out.numel() / out.shape[0]))
        return h
    for name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(mk(name, mod)))
    with torch.no_grad():
        m(torch.zeros(1, 3, 640, 640))
    for h in hooks: h.remove()
    return recs

m = build('yolo26m.yaml')
tot_p = sum(p.numel() for p in m.parameters())
print('yolo26m unfused params', tot_p, 'get_flops GFLOPs', get_flops(m, 640))
m.fuse()
tot_pf = sum(p.numel() for p in m.parameters())
print('yolo26m fused params', tot_pf, 'get_flops GFLOPs', get_flops(m, 640))
recs = profile(m, 'fused')
per = {}
for r in recs:
    d = per.setdefault(r[0], [0, 0.0, 0]); d[0] += r[6]; d[1] += r[7]; d[2] += 1
# params per top-level layer incl. non-conv
pl = {i: sum(p.numel() for p in l.parameters()) for i, l in enumerate(m.model)}
print('layer  type  params(all)  convparams  convMACs(G)  nconv')
for i, l in enumerate(m.model):
    d = per.get(i, [0, 0.0, 0])
    print(i, type(l).__name__, pl[i], d[0], round(d[1] / 1e9, 4), d[2])
tm = sum(r[7] for r in recs)
print('total conv MACs executed in forward (G):', round(tm / 1e9, 4), ' x2 =', round(2 * tm / 1e9, 3))
stem = sum(per[i][1] for i in range(0, 6)); mid = sum(per.get(i, [0, 0, 0])[1] for i in range(6, 23)); det = per[23][1]
print('MACs G stem0-5 / 6-22 / detect23:', round(stem / 1e9, 4), round(mid / 1e9, 4), round(det / 1e9, 4))
print('params stem0-5 / 6-22 / 23:', sum(pl[i] for i in range(6)), sum(pl[i] for i in range(6, 23)), pl[23])
# routed 1x1 set: layers 6-22, k=1, groups=1
r11 = [r for r in recs if 6 <= r[0] <= 22 and r[2] == 1 and r[5] == 1]
print('1x1 dense convs in 6-22:', len(r11), 'params', sum(r[6] for r in r11), 'weights-only', sum(r[3] * r[4] for r in r11), 'MACs G', round(sum(r[7] for r in r11) / 1e9, 4), 'out elems', sum(r[8] for r in r11), 'out channels', sum(r[4] for r in r11))
# block-output convs: last 1x1 conv (cv2 of C3k2 / C2PSA / SPPF) per block
for r in r11: print('  ', r[1], r[3], r[4], int(r[6]), round(r[7] / 1e9, 4))
# detect: which convs ran (one2one vs one2many)
dn = sorted(set(r[1].split('.')[2] for r in recs if r[0] == 23))
print('detect submodules that executed:', dn)
print('detect params by submodule:', {n: sum(p.numel() for p in c.parameters()) for n, c in m.model[23].named_children()})
json.dump([list(r) for r in recs], open('/data/tmp/ds-yolo/seminar3/work/agent3/recs.json', 'w'))
# weight-bank model
try:
    wb = build('yolo26m-wb.yaml')
    print('yolo26m-wb unfused params', sum(p.numel() for p in wb.parameters()))
    from ultralytics.nn.modules.moe.weight_bank import bank_modules, WeightBankRouter
    bm = bank_modules(wb)
    print('bank modules', len(bm), 'bank params', sum(sum(p.numel() for p in b.parameters() if True) for b in bm))
    print('router params', sum(sum(p.numel() for p in r.parameters()) for r in wb.modules() if isinstance(r, WeightBankRouter)))
except Exception as e:
    print('wb build failed', repr(e))
