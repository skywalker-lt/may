"""Agent 2: fused params and MACs of dense YOLO26 s/m/l/x (CPU, 2 threads) for the dense-front capacity anchor."""
import torch, torch.nn as nn
torch.set_num_threads(2)
from ultralytics.nn.tasks import DetectionModel
for s in "smlx":
    m = DetectionModel(f"yolo26{s}.yaml", ch=3, nc=80, verbose=False).eval().fuse(verbose=False)
    tot = [0]
    def f(mod, i, o): tot[0] += o.numel() * mod.kernel_size[0] * mod.kernel_size[1] * mod.in_channels // mod.groups
    hs = [mod.register_forward_hook(f) for mod in m.modules() if isinstance(mod, nn.Conv2d)]
    with torch.no_grad(): m(torch.zeros(1, 3, 640, 640))
    print(f"yolo26{s}: fused params {sum(p.numel() for p in m.parameters())/1e6:.3f} M, conv MACs {tot[0]/1e9:.3f} G")
