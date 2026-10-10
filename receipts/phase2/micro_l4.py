"""L4 micro graphs (8-layer chains, batch 1): what AP-first mechanisms cost per layer on a launch-bound GPU.
  dense_{p3,p4}      1x1 conv + SiLU (C=256 @80x80 / C=512 @40x40)
  softmoe4_{p3,p4}   per-pixel soft mixture of K=4 1x1 experts (gates from a 1x1 conv + softmax), summed, + SiLU: K x MACs
  tokentopk_p3       token top-k (25% of tokens) -> gather -> 1x1 conv -> scatter-add back (+ dense residual) + SiLU
  attn_p4            full self-attention at P4 (1600 tokens, C=512, 4 heads) + FFN, the dense control for MoE-at-P4 designs"""
import torch, torch.nn.functional as F
from torch import nn
class Dense(nn.Module):
    def __init__(s, c, n=8): super().__init__(); s.w = nn.ParameterList(nn.Parameter(torch.randn(c, c, 1, 1) * c ** -0.5) for _ in range(n))
    def forward(s, x):
        for w in s.w: x = F.silu(F.conv2d(x, w))
        return x
class SoftMoE(nn.Module):
    def __init__(s, c, n=8, k=4): super().__init__(); s.k = k; s.w = nn.ParameterList(nn.Parameter(torch.randn(k * c, c, 1, 1) * c ** -0.5) for _ in range(n)); s.g = nn.ParameterList(nn.Parameter(torch.randn(k, c, 1, 1) * c ** -0.5) for _ in range(n))
    def forward(s, x):
        b, c, h, w = x.shape
        for wk, g in zip(s.w, s.g):
            gate = F.conv2d(x, g).softmax(1)                      # [1,k,h,w]
            y = F.conv2d(x, wk).view(b, s.k, c, h, w)              # all k experts on all pixels
            x = F.silu((y * gate[:, :, None]).sum(1))
        return x
class TokenTopK(nn.Module):
    def __init__(s, c, n=8, keep=0.25): super().__init__(); s.keep = keep; s.w = nn.ParameterList(nn.Parameter(torch.randn(c, c) * c ** -0.5) for _ in range(n)); s.s = nn.ParameterList(nn.Parameter(torch.randn(c) * c ** -0.5) for _ in range(n))
    def forward(s, x):
        b, c, h, w = x.shape; t = x.flatten(2).transpose(1, 2)    # [1, hw, c]
        kk = int(h * w * s.keep)
        for wm, sv in zip(s.w, s.s):
            score = (t * sv).sum(-1)                               # [1, hw]
            idx = score.topk(kk, dim=1).indices                    # [1, kk]
            sel = torch.gather(t, 1, idx[..., None].expand(-1, -1, c))
            upd = sel @ wm
            t = F.silu(t + torch.zeros_like(t).scatter_add(1, idx[..., None].expand(-1, -1, c), upd))
        return t.transpose(1, 2).reshape(b, c, h, w)
class Attn(nn.Module):
    def __init__(s, c, n=2, heads=4): super().__init__(); s.h = heads; s.qkv = nn.ParameterList(nn.Parameter(torch.randn(3 * c, c) * c ** -0.5) for _ in range(n)); s.o = nn.ParameterList(nn.Parameter(torch.randn(c, c) * c ** -0.5) for _ in range(n)); s.f1 = nn.ParameterList(nn.Parameter(torch.randn(2 * c, c) * c ** -0.5) for _ in range(n)); s.f2 = nn.ParameterList(nn.Parameter(torch.randn(c, 2 * c) * c ** -0.5) for _ in range(n))
    def forward(s, x):
        b, c, h, w = x.shape; t = x.flatten(2).transpose(1, 2); d = c // s.h
        for qkv, o, f1, f2 in zip(s.qkv, s.o, s.f1, s.f2):
            q, k, v = (t @ qkv.T).view(b, -1, 3, s.h, d).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, -1, c)
            t = t + a @ o.T; t = t + F.silu(t @ f1.T) @ f2.T
        return t.transpose(1, 2).reshape(b, c, h, w)
specs = [("dense_p3", Dense(256), (1, 256, 80, 80)), ("softmoe4_p3", SoftMoE(256), (1, 256, 80, 80)), ("dense_p4", Dense(512), (1, 512, 40, 40)),
         ("softmoe4_p4", SoftMoE(512), (1, 512, 40, 40)), ("tokentopk_p3", TokenTopK(256), (1, 256, 80, 80)), ("attn_p4", Attn(512), (1, 512, 40, 40)), ("dense_p4_2", Dense(512, n=2), (1, 512, 40, 40))]
import onnx, collections
for name, m, shape in specs:
    torch.manual_seed(0); f = f"/data/tmp/ds-yolo/phase2/onnx/l4micro_{name}.onnx"
    torch.onnx.export(m.eval(), torch.randn(*shape), f, input_names=["images"], output_names=["output0"], opset_version=20, dynamo=False)
    print(name, dict(collections.Counter(n.op_type for n in onnx.load(f).graph.node)))
