"""Micro graphs for the T4 (local tool): 8-layer chains of conv+SiLU at 40x40, static kernels vs kernels gathered
(top-1) from a 4-kernel bank by a small router. Isolates what a runtime kernel costs per layer, for 1x1 and 3x3."""
import torch, torch.nn.functional as F
from torch import nn

class Chain(nn.Module):
    def __init__(self, c, k, n=8, bank=0):
        super().__init__()
        self.k, self.bank = k, bank
        self.w = nn.ParameterList(nn.Parameter(torch.randn(*((bank,) if bank else ()), c, c, k, k) * (c * k * k) ** -0.5) for _ in range(n))
        self.b = nn.ParameterList(nn.Parameter(torch.zeros(c)) for _ in range(n))
        self.fc = nn.Linear(c, 4)
    def forward(self, x):
        idx = self.fc(x.mean((2, 3))).argmax(1) if self.bank else None
        for w, b in zip(self.w, self.b):
            x = F.silu(F.conv2d(x, w[idx][0] if self.bank else w, b, padding=self.k // 2))
        return x

for name, c, k, bank in (("micro_1x1_static", 512, 1, 0), ("micro_1x1_bank", 512, 1, 4), ("micro_3x3_static", 256, 3, 0), ("micro_3x3_bank", 256, 3, 4)):
    torch.manual_seed(0); m = Chain(c, k, bank=bank).eval()
    f = f"/data/tmp/ds-yolo/phase1/onnx/{name}.onnx"
    torch.onnx.export(m, torch.randn(1, c, 40, 40), f, input_names=["images"], output_names=["output0"], opset_version=20, dynamo=False)
    import onnx, collections
    g = onnx.load(f).graph; print(name, dict(collections.Counter(n.op_type for n in g.node)))
