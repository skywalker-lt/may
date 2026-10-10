"""MAC count (inference graph, one2one heads only, fused) of YOLO26 task heads at 640, scales m and l.
Random init, CPU, one thread. Counts Conv2d / ConvTranspose2d MACs by forward hooks."""
import sys, torch, torch.nn as nn
sys.path.insert(0, '/data/YOLO-Master')
torch.set_num_threads(1)
from ultralytics.nn.modules.head import Detect, Segment26, Pose26, OBB26
from ultralytics.nn.modules.block import Proto26

def count(mod, x):
    macs = {}
    hooks = []
    def mk(name):
        def h(m, i, o):
            if isinstance(m, nn.Conv2d):
                k = m.kernel_size[0]*m.kernel_size[1]
                macs[name] = macs.get(name, 0) + o.numel()*m.in_channels//m.groups*k
            elif isinstance(m, nn.ConvTranspose2d):
                k = m.kernel_size[0]*m.kernel_size[1]
                macs[name] = macs.get(name, 0) + i[0].numel()*m.out_channels*k
        return h
    for n, m in mod.named_modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(mk(n)))
    with torch.no_grad():
        mod(x)
    for h in hooks: h.remove()
    return macs

def group(macs, keys):
    out = {}
    for n, v in macs.items():
        g = next((k for k in keys if n.startswith(k)), 'other')
        out[g] = out.get(g, 0) + v
    return out

for scale, ch in [('m', (256, 512, 512))]:  # yolo26-seg.yaml m and l: width 1.0, max_channels 512
    feats = [torch.randn(1, ch[0], 80, 80), torch.randn(1, ch[1], 40, 40), torch.randn(1, ch[2], 20, 20)]
    print(f'== scale {scale} ch={ch} (640 input)')
    for name, head in [('Detect', Detect(80, 1, True, ch)), ('Segment26', Segment26(80, 32, 256, 1, True, ch)),
                       ('Pose26', Pose26(80, (17, 3), 1, True, ch)), ('OBB26', OBB26(80, 1, 1, True, ch))]:
        head.eval(); head.export = True
        try:
            head.fuse()
        except Exception as e:
            print('fuse failed', name, e)
        m = count(head, [f.clone() for f in feats])
        g = group(m, ['one2one_cv2', 'one2one_cv3', 'one2one_cv4', 'cv4', 'proto.feat_refine', 'proto.feat_fuse', 'proto.cv1', 'proto.upsample', 'proto.cv2', 'proto.cv3', 'proto.semseg', 'cv2', 'cv3'])
        tot = sum(m.values())
        print(f'  {name}: total {tot/1e9:.3f} GMAC')
        for k, v in sorted(g.items(), key=lambda kv: -kv[1]):
            print(f'     {k:22s} {v/1e9:.3f} GMAC')
    # pose cv4 at level granularity
    p = Pose26(80, (17, 3), 1, True, ch); p.eval(); p.export = True; p.fuse()
    m = count(p, [f.clone() for f in feats])
    for lv in range(3):
        s = sum(v for n, v in m.items() if n.startswith(f'one2one_cv4.{lv}') or n.startswith(f'one2one_cv4_kpts.{lv}') or n.startswith(f'one2one_cv4_sigma.{lv}'))
        print(f'  pose one2one_cv4 level {lv}: {s/1e9:.3f} GMAC')
