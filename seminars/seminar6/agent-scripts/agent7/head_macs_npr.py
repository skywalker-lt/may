"""MACs of Segment26 Proto26 at hidden width 256 (released M) and 128 (null N2), 640 input, M channels; also the
stride-4 part (upsample, cv2, cv3) separately. CPU, one thread."""
import sys, torch, torch.nn as nn
sys.path.insert(0, '/data/YOLO-Master'); torch.set_num_threads(1)
from ultralytics.nn.modules.head import Segment26
ch = (256, 512, 512)
feats = [torch.randn(1, ch[0], 80, 80), torch.randn(1, ch[1], 40, 40), torch.randn(1, ch[2], 20, 20)]
for npr in (256, 128):
    h = Segment26(80, 32, npr, 1, True, ch); h.eval(); h.export = True; h.fuse()
    macs = {}
    def mk(n):
        def f(m, i, o):
            k = m.kernel_size[0] * m.kernel_size[1]
            macs[n] = (o.numel() * m.in_channels // m.groups * k) if isinstance(m, nn.Conv2d) else i[0].numel() * m.out_channels * k
        return f
    hs = [m.register_forward_hook(mk(n)) for n, m in h.named_modules() if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d))]
    with torch.no_grad(): h([f.clone() for f in feats])
    p = {k: v for k, v in macs.items() if k.startswith('proto')}
    s8 = sum(v for k, v in p.items() if any(k.startswith('proto.' + x) for x in ('feat_refine', 'feat_fuse', 'cv1')))
    s4 = sum(v for k, v in p.items() if any(k.startswith('proto.' + x) for x in ('upsample', 'cv2', 'cv3')))
    print(f'npr {npr}: Proto total {sum(p.values())/1e9:.2f} GMAC; stride-8 part {s8/1e9:.2f}; stride-4 part {s4/1e9:.2f}; cv4 {sum(v for k,v in macs.items() if "cv4" in k)/1e9:.2f}')
