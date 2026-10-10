"""Probe bypass forms on 6 val images: per-config number of detections > 0.25 and recall of full-L detections (>0.25, IoU>0.5, same class).
Forms: ident (cv2 sees [a,b,m0,m0]), zero (cv2 sees [a,b,m0,0]); C2PSA layer 10 always skips its second PSABlock."""
import sys, numpy as np, torch, torch.nn as nn, cv2
torch.set_num_threads(1)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.utils.metrics import box_iou
class Zero(nn.Module):
    def forward(self, x): return torch.zeros_like(x)
IM = [139, 285, 632, 724, 776, 785]
xs = []
for i in IM:
    im = LetterBox((640, 640), auto=False, scaleup=False)(image=cv2.imread(f'/data/datasets/coco/val2017/{i:012d}.jpg'))
    xs.append(torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255)
def build(layers, form):
    y = YOLO('/data/yolo-quant-work/weights/yolo26l.pt'); seq = y.model.model
    for i in layers: seq[i].m[1] = nn.Identity() if (form == 'ident' or i == 10) else Zero()
    return y.model.eval().float()
def dets(net):
    with torch.no_grad(): return [net(x)[0][0] for x in xs]
if True: ref = dets(build([], 'ident'))
def cmp(tag, out):
    n = []; rec = []
    for r, o in zip(ref, out):
        r = r[r[:, 4] > 0.25]; o2 = o[o[:, 4] > 0.25]; n.append(len(o2))
        if len(r) == 0: continue
        if len(o2) == 0: rec.append(0.); continue
        iou = box_iou(r[:, :4], o2[:, :4]) * (r[:, 5:6] == o2[:, 5][None])
        rec.append(float((iou.max(1).values > 0.5).float().mean()))
    print(f'{tag:28s} dets>0.25 per image {n}  recall of full-L dets {np.mean(rec):.2f}', flush=True)
if __name__ == "__main__": cmp("full L", ref)
if __name__ == "__main__":
  for form in ('ident', 'zero'):
      for tag, L in (('stem 2,4', [2, 4]), ('P3 16', [16]), ('P4/P5 6,8,10,13,19', [6, 8, 10, 13, 19]), ('all eight (B0)', [2, 4, 6, 8, 10, 13, 16, 19])):
          cmp(f'{form}: {tag}', dets(build(L, form)))
