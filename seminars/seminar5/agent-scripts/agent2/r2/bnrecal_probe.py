"""Does BatchNorm recalibration (no gradient step) rescue the untrained identity bypass of public YOLO26-L?
Recalibrate every BN's running stats (cumulative average) on N train2017 images, then rerun the 6-image probe.
Saves the recalibrated models as <name>_bnr.pt in request/ when --save."""
import os, sys, numpy as np, torch, torch.nn as nn, cv2
torch.set_num_threads(1)
sys.path.insert(0, '.')
from bypass_probe import build, dets, cmp, ref  # runs the probe's reference (full L) once
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
T = '/data/datasets/coco/images/train2017'; files = sorted(os.listdir(T)); sel = np.random.RandomState(1).permutation(len(files))[:N]
lb = LetterBox((640, 640), auto=False, scaleup=False)
batch = [torch.from_numpy(lb(image=cv2.imread(f'{T}/{files[i]}'))[..., ::-1].copy()).permute(2, 0, 1).float() / 255 for i in sel]
def recal(net):
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d): m.reset_running_stats(); m.momentum = None; m.train()
    with torch.no_grad():
        for j in range(0, N, 8): net(torch.stack(batch[j:j + 8]))  # whole net incl. Detect (its BNs too)
    net.eval(); return net
for tag, L in (('stem 2,4 (B3)', [2, 4]), ('all eight (B0)', [2, 4, 6, 8, 10, 13, 16, 19]), ('P3 16 + stem (B2)', [2, 4, 16]), ('P4/P5 + stem (B1)', [2, 4, 6, 8, 10, 13, 19])):
    net = recal(build(L, 'ident')); cmp(f'BN-recal {N}: {tag}', dets(net))
net = recal(build([], 'ident')); cmp(f'BN-recal {N}: full L (control)', dets(net))
