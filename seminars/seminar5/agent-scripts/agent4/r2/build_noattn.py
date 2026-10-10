"""Build public YOLO26-M with the attention term removed (x + attn(x) -> x, FFN kept) in layer 10 only, and in layers 10
and 22. Exports static batch-1 ONNX into work/agent4/request/. CPU, one thread."""
import os, sys, shutil, torch, torch.nn as nn, numpy as np
torch.set_num_threads(1)
from ultralytics import YOLO
from ultralytics.nn.modules.block import PSABlock
REQ = '/data/tmp/ds-yolo/seminar5/work/agent4/request'
SRC = '/data/yolo-quant-work/weights/yolo26m.pt'
class PSANoAttn(nn.Module):
    """PSABlock with the attention residual removed: x -> x + ffn(x)."""
    def __init__(self, blk):
        super().__init__(); self.ffn = blk.ffn; self.add = blk.add; assert blk.add
    def forward(self, x): return x + self.ffn(x)
def psa_sites(model):
    m = model.model.model
    a = m[10].m[0]; b = m[22].m[0][1]
    assert isinstance(a, PSABlock) and isinstance(b, PSABlock), (type(a), type(b))
    return m
def make(name, layers, imgsz):
    pt = f'{REQ}/{name}.pt'; shutil.copy(SRC, pt)
    y = YOLO(pt); m = psa_sites(y)
    if 10 in layers: m[10].m[0] = PSANoAttn(m[10].m[0])
    if 22 in layers: m[22].m[0][1] = PSANoAttn(m[22].m[0][1])
    out = y.export(format='onnx', imgsz=imgsz, batch=1, dynamic=False, half=False, simplify=True, device='cpu')
    os.remove(pt)
    print('exported', out, flush=True); return out
if __name__ == '__main__':
    for name, layers, sz in [('yolo26m_noattn10', {10}, 640), ('yolo26m_noattn10_22', {10, 22}, 640),
                             ('yolo26m_ref640', set(), 640), ('yolo26m_noattn10_22_512', {10, 22}, 512)]:
        make(name, layers, sz)
