import torch, torch.nn.functional as F
import ultralytics.nn.modules.block as blk
def forward_sdpa(self, x):
    B, _, H, W = x.shape; N = H * W
    qkv = self.qkv(x).flatten(2).transpose(1, 2)
    if self.area > 1:
        qkv = qkv.reshape(B * self.area, N // self.area, self.all_head_dim * 3); B, N, _ = qkv.shape
    q, k, v = qkv.view(B, N, self.num_heads, self.head_dim * 3).permute(0, 2, 1, 3).split([self.head_dim] * 3, dim=3)  # (B, heads, N, hd)
    x = F.scaled_dot_product_attention(q, k, v)                 # same scale hd**-0.5 as the manual path
    x = x.transpose(1, 2).reshape(B, N, self.all_head_dim)       # (B, N, C)
    v = v.transpose(1, 2).reshape(B, N, self.all_head_dim)
    if self.area > 1:
        x = x.reshape(B // self.area, N * self.area, self.all_head_dim); v = v.reshape(B // self.area, N * self.area, self.all_head_dim); B, N, _ = x.shape
    x = x.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
    v = v.reshape(B, H, W, self.all_head_dim).permute(0, 3, 1, 2).contiguous()
    return self.proj(x + self.pe(v))
blk.AAttn.forward = forward_sdpa
