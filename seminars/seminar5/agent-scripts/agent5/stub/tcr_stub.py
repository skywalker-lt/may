"""TCR (tile-choice refiner) build-feasibility stub: public YOLO26-M + router + static dispatch of k=16 of 100 P3 tiles
+ 4 window-transformer layers (random weights) + expert head + mask/concat + the end-to-end head's TopK(300).
Lowerings: 'gather' (TopK -> Gather on constant tile tables -> Gather on flattened maps), 'onehot' (TopK -> Equal/Cast
one-hot [16,100] -> MatMul on tile-major maps), 'gather_halo2' (gather + 2-token ring of L16 tokens as extra keys/values).
No ScatterElements anywhere: M's anchors in routed tiles are masked by an Expand of the one-hot ReduceMax."""
import os, sys, math, argparse, torch, torch.nn as nn, torch.nn.functional as F
torch.set_num_threads(1)
G, T3 = 10, 8          # 10x10 tiles; 8x8 P3 tokens per tile (64 px)
def tables(halo=0):
    p3 = torch.zeros(G*G, 64, dtype=torch.long); s4 = torch.zeros(G*G, 256, dtype=torch.long)
    p4 = torch.zeros(G*G, 64, dtype=torch.long); ctr = torch.zeros(G*G, 320, 2); strd = torch.zeros(320)
    ring = []
    for t in range(G*G):
        ty, tx = divmod(t, G)
        for r in range(64):
            ry, rx = divmod(r, 8); y, x = ty*8+ry, tx*8+rx
            p3[t, r] = y*80 + x; p4[t, r] = (y//2)*40 + x//2
            ctr[t, r] = torch.tensor([(x+0.5)*8, (y+0.5)*8]); strd[r] = 8
            for sy in range(2):
                for sx in range(2):
                    s4[t, r*4 + sy*2 + sx] = (2*y+sy)*160 + 2*x+sx
                    ctr[t, 64 + r*4 + sy*2 + sx] = torch.tensor([(2*x+sx+0.5)*4, (2*y+sy+0.5)*4]); strd[64 + r*4 + sy*2 + sx] = 4
        if halo:
            rr = []
            for y in range(ty*8-halo, ty*8+8+halo):
                for x in range(tx*8-halo, tx*8+8+halo):
                    if ty*8 <= y < ty*8+8 and tx*8 <= x < tx*8+8: continue
                    rr.append(y*80+x if (0 <= y < 80 and 0 <= x < 80) else 6400)   # 6400 = appended zero row
            ring.append(rr)
    return p3, s4, p4, ctr, strd, (torch.tensor(ring) if halo else None)

class WinLayer(nn.Module):
    def __init__(s, d=256, h=8, kv_extra=False):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3*d); s.proj = nn.Linear(d, d)
        s.n2 = nn.LayerNorm(d); s.f1 = nn.Linear(d, 4*d); s.f2 = nn.Linear(4*d, d)
        s.kvx = nn.Linear(d, 2*d) if kv_extra else None
    def forward(s, x, ring=None):                    # x [B,64,d], ring [B,R,d] (static context tokens)
        B, N, d = x.shape; q, k, v = s.qkv(s.n1(x)).chunk(3, -1)
        if ring is not None:
            kr, vr = s.kvx(ring).chunk(2, -1); k = torch.cat([k, kr], 1); v = torch.cat([v, vr], 1)
        sh = lambda t: t.reshape(B, t.shape[1], s.h, d//s.h).transpose(1, 2)
        a = F.scaled_dot_product_attention(sh(q), sh(k), sh(v)).transpose(1, 2).reshape(B, N, d)
        x = x + s.proj(a)
        return x + s.f2(F.silu(s.f1(s.n2(x))))

class TCRStub(nn.Module):
    def __init__(s, det, k=16, nl=4, d=256, lowering='gather', mode='tcr'):
        super().__init__(); s.layers = det.model[:-1]; s.det = det.model[-1]; s.k = k; s.low = lowering; s.mode = mode
        s.halo = 2 if lowering == 'gather_halo2' else 0
        p3, s4, p4, ctr, strd, ring = tables(s.halo)
        for n, t in dict(p3=p3, s4=s4, p4=p4, ctr=ctr.reshape(G*G, 640), strd=strd).items(): s.register_buffer(n, t)
        if ring is not None: s.register_buffer('ring', ring)
        s.register_buffer('ar', torch.arange(G*G))
        s.fuse = nn.Linear(256 + 1024 + 512, d)
        s.blocks = nn.ModuleList(WinLayer(d, 8, kv_extra=s.halo > 0) for _ in range(nl))
        s.ringp = nn.Linear(256, d) if s.halo else None
        s.head = nn.Linear(d, 5 * 84)               # per token: 1 stride-8 anchor + 4 stride-4 sub-anchors, 4 box + 80 cls
    def backbone(s, x):
        y = []
        for m in s.layers:
            if m.f != -1: x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x); y.append(x)
        return y
    def forward(s, im):
        y = s.backbone(im); feats = [y[16], y[19], y[22]]
        pr = s.det.forward_head(feats, **s.det.one2one)
        dec = s.det._inference(pr)                  # [1, 84, 8400]: xyxy (px) + sigmoid scores
        if s.mode == 'base':
            return s.det.postprocess(dec.permute(0, 2, 1))
        sc = dec[:, 4:]                              # [1,80,8400]
        p = sc.max(1)[0]                             # [1,8400]
        u = p * (1 - p) * (p > 0.01).float()
        u3 = u[:, :6400].reshape(G, 8, G, 8).sum((1, 3)); u4 = u[:, 6400:8000].reshape(G, 4, G, 4).sum((1, 3))
        u5 = u[:, 8000:].reshape(G, 2, G, 2).sum((1, 3))
        tile_u = (u3 + u4 + u5).reshape(G*G)
        idx = tile_u.topk(s.k)[1]                    # [16] static TopK
        L16, L2, L19 = y[16][0], y[2][0], y[19][0]   # [256,80,80], [256,160,160], [512,40,40]
        oh = (s.ar[None, :] == idx[:, None]).float() # [16,100]   (used for the mask in every lowering)
        if s.low.startswith('gather'):
            f3 = L16.flatten(1).t(); f2 = L2.flatten(1).t(); f4 = L19.flatten(1).t()
            t3 = f3[s.p3[idx].flatten()].reshape(s.k, 64, 256)
            t2 = f2[s.s4[idx].flatten()].reshape(s.k, 64, 1024)
            t4 = f4[s.p4[idx].flatten()].reshape(s.k, 64, 512)
            ctr = s.ctr[idx].reshape(s.k, 320, 2)
        else:
            m3 = L16.reshape(256, G, 8, G, 8).permute(1, 3, 2, 4, 0).reshape(G*G, 64*256)
            m2 = L2.reshape(256, G, 8, 2, G, 8, 2).permute(1, 4, 2, 5, 3, 6, 0).reshape(G*G, 64*1024)
            m4 = L19.reshape(512, G, 4, G, 4).permute(1, 3, 2, 4, 0).reshape(G*G, 16*512)
            t3 = (oh @ m3).reshape(s.k, 64, 256); t2 = (oh @ m2).reshape(s.k, 64, 1024)
            t4 = (oh @ m4).reshape(s.k, 4, 1, 4, 1, 512).expand(s.k, 4, 2, 4, 2, 512).reshape(s.k, 64, 512)
            ctr = (oh @ s.ctr).reshape(s.k, 320, 2)
        x = s.fuse(torch.cat([t3, t2, t4], -1))      # [16,64,256]
        ring = None
        if s.halo:
            f3z = torch.cat([L16.flatten(1).t(), torch.zeros(1, 256, dtype=L16.dtype)], 0)
            ring = s.ringp(f3z[s.ring[idx].flatten()].reshape(s.k, -1, 256))
        for b in s.blocks: x = b(x, ring)
        o = s.head(x).reshape(s.k, 64, 5, 84)
        o = torch.cat([o[:, :, :1].reshape(s.k, 64, 84), o[:, :, 1:].reshape(s.k, 256, 84)], 1)   # [16,320,84]
        lt, rb = o[..., :2], o[..., 2:4]; st = s.strd[None, :, None]
        box = torch.cat([ctr - lt * st, ctr + rb * st], -1)          # xyxy px
        esc = o[..., 4:].sigmoid()
        if s.mode == 'off': esc = esc * 0                            # check mode: expert never selected
        ebox = box.reshape(1, s.k*320, 4).permute(0, 2, 1); esc = esc.reshape(1, s.k*320, 80).permute(0, 2, 1)
        inm = oh.max(0)[0].reshape(G, 1, G, 1).expand(G, 8, G, 8).reshape(1, 1, 6400)
        keep = torch.cat([1 - inm, torch.ones(1, 1, 2000, dtype=inm.dtype)], -1)
        if s.mode == 'off': keep = keep * 0 + 1
        allp = torch.cat([torch.cat([dec[:, :4], ebox], -1), torch.cat([sc * keep, esc], -1)], 1)  # [1,84,13520]
        return s.det.postprocess(allp.permute(0, 2, 1))

def build(lowering, mode='tcr', k=16):
    from ultralytics import YOLO
    from ultralytics.nn.modules.head import Detect
    det = YOLO('/data/yolo-quant-work/weights/yolo26m.pt').model.float().eval().fuse()
    for m in det.modules():
        if isinstance(m, Detect): m.export = True; m.format = 'onnx'; m.dynamic = False; m.max_det = 300; m.shape = None
    torch.manual_seed(0)
    s = TCRStub(det, k=k, lowering=lowering, mode=mode).eval()
    for p in s.parameters(): p.requires_grad = False
    return s

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--lowering', default='gather'); ap.add_argument('--mode', default='tcr')
    ap.add_argument('--out', required=True); a = ap.parse_args()
    s = build(a.lowering, a.mode); im = torch.zeros(1, 3, 640, 640)
    with torch.no_grad(): ref = s(im)
    torch.onnx.export(s, im, a.out, opset_version=18, do_constant_folding=True, input_names=['images'],
                      output_names=['output0'], dynamo=False)
    print('exported', a.out, tuple(ref.shape))
