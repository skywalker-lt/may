"""Deployment-graph prototypes for residual experts (local tool; timing prototypes, deltas are small random values).
Every variant keeps the shared kernel W of YOLO26-M and adds a routed delta chosen by the same router (top-2 of 4):
  res_full     kernel = W + sum g_i D_i, full-rank deltas mixed in weight space -> one runtime-kernel conv per layer
  res_scale    y = act(s_g * conv_W(x) + b_g): per-channel scale and shift deltas (D_i = diag(s_i) W), conv stays static
  res_bias     y = act(conv_W(x) + b_g): per-channel shift only, conv stays static
  res_lowrank  y = act(conv_[W;A](x)[:c2] + B_g * conv_[W;A](x)[c2:]): rank-r deltas D_i = B_i A, shared A merged
               into the static conv as r extra output channels; only the small B_g (c2 x r) is a runtime kernel
The routed 39 layers and the router input are those of the weight bank (layers 6-22, router on the input of layer 6)."""
import collections, shutil, sys
import cv2, numpy as np, onnx, torch, torch.nn.functional as F
from torch import nn
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.moe.weight_bank import WeightBankRouter, is_bankable

OUT = "/data/tmp/ds-yolo/phase1/onnx"; DENSE = "/data/yolo-quant-work/weights/yolo26m.pt"; E, K, R = 4, 2, 16

class Gate:
    def __init__(self): self.idx = self.w = self.vec = None; self.tables = []; self.off = []
    def add(self, table):  # table [E, n] -> offset into the concatenated table
        o = sum(t.shape[1] for t in self.tables); self.tables.append(table.detach()); return o
    def flat(self): return torch.cat(self.tables, 1)

class Proto(nn.Conv2d):
    """Replaces the fused nn.Conv2d inside a Conv block (the block applies the activation afterwards)."""
    def setup(self, mode, gate, router):
        self.mode, self.gate, self.router = mode, gate, router
        c2, c1 = self.out_channels, self.in_channels; g = torch.Generator().manual_seed(c1 * 131 + c2)
        if mode == "full":
            self.n = c2 * c1; t = 0.01 * self.weight.std() * torch.randn(E, self.n, generator=g)
        elif mode == "scale":
            self.n = 2 * c2; t = torch.cat([1 + 0.05 * torch.randn(E, c2, generator=g), 0.05 * torch.randn(E, c2, generator=g)], 1)
        elif mode == "bias":
            self.n = c2; t = 0.05 * torch.randn(E, c2, generator=g)
        elif mode == "lowrank":
            self.n = c2 * R; t = 0.01 * torch.randn(E, self.n, generator=g)
            a = self.weight.detach().std() * torch.randn(R, c1, 1, 1, generator=g)
            self.weight = nn.Parameter(torch.cat([self.weight.data, a], 0)); self.bias = nn.Parameter(torch.cat([self.bias.data, torch.zeros(R)]))
        self.o = gate.add(t)
    def forward(self, x):
        gate = self.gate
        if self.router is not None:
            vals, idx = self.router.logits(x).topk(K, dim=1)
            gate.vec = vals.softmax(1) @ gate.flat()[idx[0]]  # one gather and one mix over the concatenated delta table
        d = gate.vec[0, self.o : self.o + self.n]
        c2 = self.out_channels
        if self.mode == "full":
            return F.conv2d(x, self.weight + d.view(c2, self.in_channels, 1, 1), self.bias)
        y = F.conv2d(x, self.weight, self.bias)
        if self.mode == "scale":
            return y * d[:c2].view(1, c2, 1, 1) + d[c2:].view(1, c2, 1, 1)
        if self.mode == "bias":
            return y + d.view(1, c2, 1, 1)
        return y[:, :c2] + F.conv2d(y[:, c2:], d.view(c2, R, 1, 1))

im = LetterBox((640, 640), auto=False)(image=cv2.imread("/data/datasets/coco/images/val2017/000000000139.jpg"))
x = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]
for arg in sys.argv[1:] or ("full", "scale", "bias", "lowrank"):
    mode, subset = arg.split("@") if "@" in arg else (arg, "all")  # "@out": only the output conv (cv2) of each block
    torch.manual_seed(0)
    y = YOLO(DENSE); m = y.model.float().eval().fuse(verbose=False)
    gate, n = Gate(), 0
    for layer in list(m.model)[6:23]:
        for blk in ([layer.cv2] if subset == "out" and hasattr(layer, "cv2") else [] if subset == "out" else layer.modules()):
            if type(blk) is Conv and is_bankable(blk.conv):
                c = blk.conv; p = Proto(c.in_channels, c.out_channels, 1, bias=True)
                p.weight.data.copy_(c.weight.data); p.bias.data.copy_(c.bias.data)
                p.setup(mode, gate, WeightBankRouter(c.in_channels, E, K) if n == 0 else None); blk.conv = p; n += 1
    with torch.no_grad(): m(x)
    gate.vec = None
    f = y.export(format="onnx", imgsz=640, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
    dst = f"{OUT}/res_{mode}{'' if subset == 'all' else '_' + subset}.onnx"; shutil.move(f, dst)
    g = onnx.load(dst).graph; ops = collections.Counter(nd.op_type for nd in g.node); init = {i.name for i in g.initializer}
    dyn = sum(1 for nd in g.node if nd.op_type == "Conv" and nd.input[1] not in init)
    print(f"res_{mode}@{subset}: routed {n} layers, delta table {tuple(gate.flat().shape)} | Conv {ops['Conv']} (kernel-as-input {dyn}) MatMul {ops['MatMul']} Gather {ops['Gather']} "
          f"Slice {ops['Slice']} Mul {ops['Mul']} Add {ops['Add']} If {ops['If']} | {round(sum(len(i.raw_data) for i in g.initializer) / 1e6, 1)} MB", flush=True)
