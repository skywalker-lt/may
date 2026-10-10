"""Agent 1 round 2: step-0 router, image vs mirror, on 1000 val2017 images disjoint from agent 4's 250. CPU, 2 threads."""
import glob, cv2, numpy as np, torch
torch.set_num_threads(2)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.moe.weight_bank import bank_modules
wb = YOLO("/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt").model.float().eval()
router = bank_modules(wb)[0].router
files = sorted(glob.glob("/data/datasets/coco/images/val2017/*.jpg"))[2::5][:1000]
lb = LetterBox((640, 640), auto=False)
def feat(x):
    for i in range(6): x = wb.model[i](x)
    return x
L, LF = [], []
with torch.no_grad():
    for f in files:
        im = lb(image=cv2.imread(f)); x = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]
        L.append(router.logits(feat(x))[0]); LF.append(router.logits(feat(x.flip(3)))[0])
L, LF = torch.stack(L), torch.stack(LF)
torch.save((L, LF), "/data/tmp/ds-yolo/seminar3/work/agent1/flip_logits.pt")
t = L.topk(2, 1); mg = (t.values[:, 0] - t.values[:, 1]).numpy()
fl = (L.argmax(1) != LF.argmax(1)).numpy()
c = torch.bincount(L.argmax(1), minlength=4).float(); p = c / c.sum()
print("N", len(files), "argmax counts", c.tolist(), "entropy bits %.3f" % float(-(p * p.log2()).sum()))
print("margin median %.4f frac<0.01 %.3f frac<0.05 %.3f" % (np.median(mg), (mg < .01).mean(), (mg < .05).mean()))
print("top1 flips under hflip %.3f; rms dlogit %.4f; median |margin| of flipped %.4f; flips among margin>0.05 %.3f, among margin>0.1 %.3f" % (fl.mean(), float((L - LF).pow(2).mean().sqrt()), np.median(mg[fl]), fl[mg > .05].mean(), fl[mg > .1].mean()))
s = lambda A: [frozenset(r) for r in A.topk(2, 1).indices.tolist()]
print("same top-2 set under hflip %.3f" % np.mean([a == b for a, b in zip(s(L), s(LF))]))
