"""Agent 1 round 4: natural vs balanced k-means (k=4) shares on the pooled layer-5 feature, and mirror agreement
of the k-means label. 600 val2017 images, CPU, 2 threads. Statistic only, no latency."""
import glob, cv2, numpy as np, torch
torch.set_num_threads(2)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox

class KMeans:
    def __init__(self, k, n_init=10, random_state=0): self.k, self.n, self.rs = k, n_init, random_state
    def fit(self, Z):
        rng = np.random.default_rng(self.rs); best = None
        for _ in range(self.n):
            C = Z[rng.choice(len(Z), self.k, replace=False)].copy()
            for _ in range(100):
                lab = ((Z[:, None] - C[None]) ** 2).sum(-1).argmin(1)
                Cn = np.stack([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(self.k)])
                if np.allclose(Cn, C): break
                C = Cn
            inertia = ((Z - C[lab]) ** 2).sum()
            if best is None or inertia < best[0]: best = (inertia, C, lab)
        _, self.cluster_centers_, self.labels_ = best; return self
    def predict(self, Z): return ((Z[:, None] - self.cluster_centers_[None]) ** 2).sum(-1).argmin(1)
from scipy.optimize import linear_sum_assignment
N = 600
net = YOLO("/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt").model.float().eval()
stem = list(net.model[:6]); assert all(m.f == -1 for m in stem)
files = sorted(glob.glob("/data/datasets/coco/images/val2017/*.jpg"))[3::8][:N]
lb = LetterBox((640, 640), auto=False)
F, FM = [], []
with torch.no_grad():
    for f in files:
        im = lb(image=cv2.imread(f)); x = torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]
        for xx, out in ((x, F), (x.flip(-1), FM)):
            y = xx
            for m in stem: y = m(y)
            out.append(y.mean((2, 3))[0].numpy())
F, FM = np.stack(F), np.stack(FM)
mu, sd = F.mean(0), F.std(0) + 1e-6
Z, ZM = (F - mu) / sd, (FM - mu) / sd
print("images", len(F), "dim", F.shape[1])
for seed in range(3):
    km = KMeans(4, n_init=10, random_state=seed).fit(Z)
    lab = km.labels_; shares = np.sort(np.bincount(lab, minlength=4) / len(lab))[::-1]
    mlab = km.predict(ZM)
    d = ((Z[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    ds = np.sort(d, 1); rel = (ds[:, 1] - ds[:, 0]) / ds[:, 0]
    # balanced assignment: each centroid replicated N/4 times
    cost = np.repeat(d, len(Z) // 4, axis=1)
    r, c = linear_sum_assignment(cost); bal = np.empty(len(Z), int); bal[r] = c // (len(Z) // 4)
    dm = ((ZM[:, None, :] - km.cluster_centers_[None]) ** 2).sum(-1)
    print(f"seed {seed}: natural shares {np.round(shares, 3)}; mirror keeps natural label {np.mean(mlab == lab):.3f}; "
          f"balanced differs from natural on {np.mean(bal != lab):.3f}; inertia ratio balanced/natural "
          f"{d[np.arange(len(Z)), bal].sum() / d[np.arange(len(Z)), lab].sum():.4f}; "
          f"median relative centroid margin {np.median(rel):.3f}")
# random-projection null: how mirror-stable is the feature itself
print("mean |z - z_mirror| / mean |z - z_other|:",
      round(float(np.linalg.norm(Z - ZM, axis=1).mean() / np.linalg.norm(Z - np.roll(Z, 1, 0), axis=1).mean()), 4))
