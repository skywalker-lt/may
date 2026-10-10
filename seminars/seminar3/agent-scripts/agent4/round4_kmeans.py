"""Round 4 check for vote V3: natural against balanced k-means (K=4) on the pooled layer-5 feature of public YOLO26-M.

Reports cluster shares, variance explained, and agreement of the cluster label between an image and its mirror.
CPU, 2 threads, N val2017 images (every 5000//N-th file).
"""
import glob, sys
import cv2, numpy as np, torch
from scipy.optimize import linear_sum_assignment
torch.set_num_threads(2)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

N = int(sys.argv[1]); K = 4
wb = YOLO("/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt").model.float().eval()
files = sorted(glob.glob("/data/datasets/coco/images/val2017/*.jpg"))[:: 5000 // N][:N]
lb = LetterBox((640, 640), auto=False)
def feat(x):
    for i in range(6): x = wb.model[i](x)
    return x.mean((2, 3))[0].numpy()
F, FM = [], []
with torch.no_grad():
    for f in files:
        im = lb(image=cv2.imread(f)); x = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]
        F.append(feat(x)); FM.append(feat(x.flip(3)))
F, FM = np.stack(F), np.stack(FM)
mu, sd = F.mean(0), F.std(0) + 1e-6
Z, ZM = (F - mu) / sd, (FM - mu) / sd
tss = (Z ** 2).sum()
def d2(Z, C): return ((Z[:, None] - C[None]) ** 2).sum(-1)
def kmeans(seed, balanced):
    rng = np.random.default_rng(seed); C = Z[rng.choice(len(Z), K, replace=False)]
    for _ in range(40):
        D = d2(Z, C)
        if balanced:
            r, c = linear_sum_assignment(np.tile(D, (1, int(np.ceil(len(Z) / K)))))  # column j is centroid j % K
            lab = np.empty(len(Z), int); lab[r] = c % K
        else:
            lab = D.argmin(1)
        C = np.stack([Z[lab == k].mean(0) if (lab == k).any() else C[k] for k in range(K)])
    return (d2(Z, C)[np.arange(len(Z)), lab]).sum(), lab, C
for balanced in (False, True):
    runs = [kmeans(s, balanced) for s in range(8 if not balanced else 3)]
    wss, lab, C = min(runs, key=lambda r: r[0])
    shares = np.sort(np.bincount(lab, minlength=K) / len(Z))[::-1]
    mins = [np.bincount(r[1], minlength=K).min() / len(Z) for r in runs]
    labm = d2(ZM, C).argmin(1); nat = d2(Z, C).argmin(1)
    D = np.sort(d2(Z, C), 1); marg = np.sqrt(D[:, 1]) - np.sqrt(D[:, 0])
    print(f"{'balanced' if balanced else 'natural '} K=4 N={len(Z)} dim={Z.shape[1]}: shares {np.round(shares, 3)}; "
          f"smallest share over restarts {min(mins):.3f}-{max(mins):.3f}; variance explained {1 - wss / tss:.4f}; "
          f"nearest-centroid label, image vs mirror: {np.mean(nat == labm):.4f}; "
          f"images whose assigned centroid is not the nearest: {np.mean(nat != lab):.3f}; "
          f"relative centroid margin median {np.median(marg / np.sqrt(D[:, 0])):.3f}")
print(f"pooled feature, image vs mirror: relative L2 change median {np.median(np.linalg.norm(Z - ZM, axis=1) / np.linalg.norm(Z, axis=1)):.4f}")
