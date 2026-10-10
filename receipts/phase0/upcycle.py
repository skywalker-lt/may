"""Upcycle a dense YOLO26 checkpoint into a weight-bank checkpoint (local tool).
Experts = W + eps_i (sum zero, 1% of |W|); router input statistics calibrated on N images; the checkpoint is then
checked for the step-0 router signal on val images (real images vs a pixel-shuffled control)."""
import argparse, collections, glob, itertools, os
import cv2, numpy as np, torch
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from ultralytics.nn.modules.moe.weight_bank import bank_modules, convert_to_weight_bank

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--calib", default="/data/datasets/coco/images/train2017"); ap.add_argument("--n-calib", type=int, default=512)
ap.add_argument("--val", default="/data/datasets/coco/images/val2017"); ap.add_argument("--n-val", type=int, default=500)
ap.add_argument("--experts", type=int, default=4); ap.add_argument("--top-k", type=int, default=2)
ap.add_argument("--eps", type=float, default=0.01); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
lb = LetterBox((640, 640), auto=False)

def load(files, shuffle_pixels=False, flip=False):
    for f in files:
        im = lb(image=cv2.imread(f))
        if flip:  # label-preserving change of the same image
            im = im[:, ::-1]
        if shuffle_pixels:  # same colour histogram, no structure
            flat = im.reshape(-1, 3); im = flat[rng.permutation(len(flat))].reshape(im.shape)
        yield torch.from_numpy(im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]

def pick(d, n):
    allf = sorted(glob.glob(os.path.join(d, "*.jpg"))); return allf[:: max(1, len(allf) // n)][:n]

ck = torch.load(a.src, map_location="cpu", weights_only=False)
model = (ck.get("ema") or ck["model"]).float().eval()
dense_params = sum(p.numel() for p in model.parameters())
state = convert_to_weight_bank(model.model, 6, 22, experts=a.experts, top_k=a.top_k, eps=a.eps)
model.yaml["weight_bank"] = dict(layers=[6, 22], experts=a.experts, top_k=a.top_k, hidden=64, eps=0.0)
owner = bank_modules(model)[0]; router = owner.router
print(f"converted {len(state.members)} convs; params {dense_params:,} -> {sum(p.numel() for p in model.parameters()):,}")

# router input statistics from N calibration images (stop the forward right after the router input is seen)
class Stop(Exception): pass
pooled = []
def grab(m, args):
    pooled.append(args[0].float().mean((2, 3))); raise Stop
h = owner.register_forward_pre_hook(grab)
with torch.no_grad():
    for x in load(pick(a.calib, a.n_calib)):
        try: model(x)
        except Stop: pass
h.remove()
P = torch.cat(pooled)
router.norm.running_mean.copy_(P.mean(0)); router.norm.running_var.copy_(P.var(0, unbiased=False))
cv = (P.std(0) / P.mean(0).abs().clamp_min(1e-6)).median()
print(f"router input: {P.shape[0]} images x {P.shape[1]} channels; median across-image std / |mean| per channel = {cv:.3f}")

def router_logits(files, shuffle_pixels=False, flip=False):
    out = []
    h = owner.register_forward_pre_hook(lambda m, args: (out.append(router.logits(args[0])), (_ for _ in ()).throw(Stop))[0])
    with torch.no_grad():
        for x in load(files, shuffle_pixels, flip):
            try: model(x)
            except Stop: pass
    h.remove(); return torch.cat(out)

vf = pick(a.val, a.n_val)
L = router_logits(vf); S = router_logits(vf, shuffle_pixels=True)
def report(tag, L):
    std = L.std(0); spread = L.mean(0).std()
    sets = collections.Counter(tuple(sorted(r)) for r in L.topk(a.top_k, 1).indices.tolist())
    top1 = collections.Counter(L.argmax(1).tolist())
    print(f"{tag}: logit std across {len(L)} images per expert {[round(float(v), 3) for v in std]} (mean {std.mean():.3f}); "
          f"spread of mean logits {spread:.3f}; ratio {std.mean() / spread:.2f}")
    print(f"    top-{a.top_k} sets: {len(sets)} distinct, shares {[(k, round(v / len(L), 3)) for k, v in sets.most_common()]}")
    print(f"    top-1 shares {sorted((k, round(v / len(L), 3)) for k, v in top1.items())}")
    return float(std.mean())
s_real = report("real images     ", L); s_shuf = report("pixel-shuffled  ", S)
print(f"gate C12: logit std {s_real:.3f} >= 0.1 -> {'PASS' if s_real >= 0.1 else 'FAIL'}; pixel-shuffled / real = {s_shuf / s_real:.2f}")
Fl = router_logits(vf, flip=True)
within = ((L - Fl) / 2 ** 0.5).std(0).mean()  # std of one image's logit under a horizontal flip
agree = sum(set(x) == set(y) for x, y in zip(L.topk(a.top_k, 1).indices.tolist(), Fl.topk(a.top_k, 1).indices.tolist())) / len(L)
print(f"within-image control (horizontal flip): within-image logit std {within:.3f} vs between-image {s_real:.3f}; "
      f"ratio {within / s_real:.2f} (signal if well below 1); same top-{a.top_k} set for image and its flip on {agree:.1%}")

ck_out = {k: v for k, v in ck.items() if k not in ("model", "ema", "optimizer", "scaler", "updates")}
ck_out.update(model=model, ema=None, optimizer=None, updates=None, epoch=-1, best_fitness=None)
torch.save(ck_out, a.out); print("saved", a.out, os.path.getsize(a.out))
