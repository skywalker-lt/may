"""YOLO26-M with one extra C2PSA-style block on P4 after layer 13 (cv1 512->512, PSABlock on 256 ch, 4 heads, key dim 32, FFN 2x,
cv2 512->512), residual: y = C3k2_13(x); y + blk(y). cv2's BN gamma and beta are zero, so the block outputs exactly 0 and the
model equals public YOLO26-M (epoch-0 identity of the round-1 construct). Exported at 512 (the shrink branch) and 640."""
import os, shutil, torch, torch.nn as nn
torch.set_num_threads(1); torch.manual_seed(0)
from ultralytics import YOLO
from ultralytics.nn.modules.block import C2PSA
REQ='/data/tmp/ds-yolo/seminar5/work/agent4/request'; SRC='/data/yolo-quant-work/weights/yolo26m.pt'
class P4Attn(nn.Module):
    def __init__(self, inner):
        super().__init__(); self.inner=inner; self.blk=C2PSA(512,512,1,0.5)
        nn.init.zeros_(self.blk.cv2.bn.weight); nn.init.zeros_(self.blk.cv2.bn.bias)
        for a in ('f','i','type','np'): setattr(self,a,getattr(inner,a))
    def forward(self, x):
        y=self.inner(x); return y+self.blk(y)
def make(name, sz):
    pt=f'{REQ}/{name}.pt'; shutil.copy(SRC,pt); y=YOLO(pt); m=y.model.model
    m[13]=P4Attn(m[13]); y.model.eval()
    for p in y.model.parameters(): p.requires_grad_(False)
    x=torch.rand(1,3,sz,sz); 
    out=y.export(format='onnx',imgsz=sz,batch=1,dynamic=False,half=False,simplify=True,device='cpu'); os.remove(pt); print('exported',out,flush=True)
if __name__=='__main__':
    make('yolo26m_p4attn0_512',512); make('yolo26m_p4attn0_640',640)
