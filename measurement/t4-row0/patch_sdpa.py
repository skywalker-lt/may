import torch, torch.nn.functional as F
import ultralytics.nn.modules.block as blk
def forward_sdpa(self, x):
    B, C, H, W = x.shape; N = H * W
    qk = self.qk(x).flatten(2).transpose(1, 2); v = self.v(x); pp = self.pe(v); v = v.flatten(2).transpose(1, 2)
    if self.area > 1:
        qk = qk.reshape(B * self.area, N // self.area, C * 2); v = v.reshape(B * self.area, N // self.area, C); B, N, _ = qk.shape
    q, k = qk.split([C, C], dim=2)
    q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
    x = F.scaled_dot_product_attention(q, k, v)
    x = x.transpose(1, 2).reshape(B, N, C)
    if self.area > 1:
        x = x.reshape(B // self.area, N * self.area, C); B, N, _ = x.shape
    x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
    return self.proj(x + pp)
blk.AAttn.forward = forward_sdpa
