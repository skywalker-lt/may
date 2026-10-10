"""Agent 4: step-0 router margins on val2017 (CPU, correctness-type statistic only, no latency).
Top-1/top-2 logit margin, argmax balance, argmax flips under horizontal flip and under fp16 rounding of the router input."""
import glob, sys, cv2, numpy as np, torch
torch.set_num_threads(2)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.moe.weight_bank import bank_modules
N = int(sys.argv[1])
wb = YOLO("/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt").model.float().eval()
router = bank_modules(wb)[0].router
files = sorted(glob.glob("/data/datasets/coco/images/val2017/*.jpg"))[:: 5000 // N][:N]
lb = LetterBox((640, 640), auto=False)
def feat(x):
    for i in range(6): x = wb.model[i](x)
    return x
L, LF, LH = [], [], []
with torch.no_grad():
    for f in files:
        im = lb(image=cv2.imread(f)); x = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]
        a = feat(x); b = feat(x.flip(3))
        L.append(router.logits(a)[0]); LF.append(router.logits(b)[0]); LH.append(router.logits(a.half().float())[0])
L, LF, LH = torch.stack(L), torch.stack(LF), torch.stack(LH)
t = L.topk(2, 1).values; mg = (t[:, 0] - t[:, 1]).numpy()
print("N", N, "logit std between images (mean over experts)", float(L.std(0).mean()))
print("argmax counts", torch.bincount(L.argmax(1), minlength=4).tolist())
print("top2 pair counts", np.unique(np.sort(L.topk(2, 1).indices.numpy(), 1), axis=0, return_counts=True)[1].tolist())
print("margin top1-top2: median %.4f p10 %.4f p25 %.4f; frac<0.01 %.3f frac<0.02 %.3f frac<0.05 %.3f" % (np.median(mg), np.percentile(mg, 10), np.percentile(mg, 25), (mg < .01).mean(), (mg < .02).mean(), (mg < .05).mean()))
print("argmax flips under hflip %.3f; logit rms diff %.4f" % (float((L.argmax(1) != LF.argmax(1)).float().mean()), float((L - LF).pow(2).mean().sqrt())))
print("argmax flips under fp16 rounding of router input %.4f; max |dlogit| %.5f" % (float((L.argmax(1) != LH.argmax(1)).float().mean()), float((L - LH).abs().max())))
