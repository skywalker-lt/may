"""Forced-branch models from a weight-bank checkpoint (seminar 5, agent 6 request).

A forced branch is the network with every gate fixed to one expert for every image. Because each banked layer is a 1x1
convolution whose kernel is the gated sum of its bank (weight_bank.py), fixing the gates is the same as replacing every
BankConv2d by a plain nn.Conv2d holding that expert's kernel (and the bank's shared bias). The result is an ordinary
YOLO26-M DetectionModel with no weight-bank module, saved as a standard .pt that YOLO(pt), the CLI and
scripts/ds_yolo/dump_batch.sh (spec line "<name> <pt> <size> o2o") handle like any dense checkpoint.

Outputs in --out:
  <tag>_e{i}.pt            expert i forced on every image (i = 0..E-1)          [always]
  <tag>_pair{i}{j}.pt      hard pair (i, j), kernels averaged                     [--pairs, the six-pair arm's branches]
  <tag>_merged.pt          uniform average of all E kernels (the merged constant) [--merged]
  spec_<tag>.txt           dump_batch.sh spec lines for all of the above
  route_<tag>.json         {image_id: [top-k expert indices]} of the checkpoint's own router on val2017  [--route-dir]
  check_<tag>.log          equivalence check: banked model with gates forced to one-hot(i) vs the forced-branch model

usage (pod):  PYTHONPATH=/data/YOLO-Master python make_forced_branches.py --ckpt /training_data/runs80/top180/weights/last.pt \
                  --tag top180_ep20 --out /data/weights/forced --pairs --merged --route-dir /data/datasets/coco/images/val2017 --device 0
smoke (CPU):  OMP_NUM_THREADS=1 PYTHONPATH=/data/YOLO-Master python make_forced_branches.py \
                  --ckpt /data/tmp/ds-yolo/weights/yolo26m-objv1-wb-init.pt --tag init --out <dir> --perturb 0.05 --merged
"""

import argparse
import copy
import glob
import itertools
import json
import os
import time

import torch
from torch import nn

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.moe.weight_bank import BankConv2d, bank_modules


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = (ck.get("ema") or ck["model"]).float().eval()
    for b in bank_modules(m):  # checkpoints older than GateState.hard
        if not hasattr(b.state, "hard"):
            b.state.hard = False
    return ck, m


def force(model, sel):
    """Deep copy of model with every BankConv2d replaced by a plain 1x1 conv = mean of the selected experts' kernels."""
    m = copy.deepcopy(model)
    n = 0
    for blk in m.modules():
        if type(blk) is Conv and isinstance(blk.conv, BankConv2d):
            b = blk.conv
            w = torch.stack([b.experts[i].detach() for i in sel]).float().mean(0)
            conv = nn.Conv2d(b.in_channels, b.out_channels, 1, bias=b.bias is not None)
            with torch.no_grad():
                conv.weight.copy_(w)
                if b.bias is not None:
                    conv.bias.copy_(b.bias.detach().float())
            blk.conv = conv
            n += 1
    assert not bank_modules(m), "a BankConv2d survived outside a Conv block"
    m.yaml = {k: v for k, v in m.yaml.items() if k != "weight_bank"}  # a rebuild from yaml must not re-bank
    m.yaml["forced_branch"] = list(sel)
    return m, n


def set_forced_gates(model, sel):
    """Make the banked model's router emit fixed gates (equal weights on sel) for every image; returns an undo."""
    r = bank_modules(model)[0].router
    E = bank_modules(model)[0].state.num_experts

    def gates(logits, top_k, hard=False):
        g = torch.zeros(logits.shape[0], E, dtype=logits.dtype, device=logits.device)
        g[:, list(sel)] = 1.0 / len(sel)
        return g

    r.gates_from_logits = gates
    return lambda: r.__dict__.pop("gates_from_logits", None)


def outputs(model, x):
    """The three pyramid maps entering the Detect head (after every banked layer), flattened per image. Compared
    instead of the end-to-end [B, 300, 6] output, whose TopK order flips on near-tied scores."""
    det = model.model[-1]
    cap = {}
    h = det.register_forward_pre_hook(lambda m, inp: cap.__setitem__("x", [t.detach().float().clone() for t in inp[0]]))
    try:
        with torch.no_grad():
            model(x)
    finally:
        h.remove()
    return torch.cat([t.flatten(1) for t in cap["x"]], 1)


def save(ck, m, path, sel, src):
    out = {k: v for k, v in ck.items() if k not in ("model", "ema", "optimizer", "scaler", "updates")}
    out.update(model=copy.deepcopy(m).half(), ema=None, optimizer=None, updates=None, forced_branch=dict(experts=list(sel), source=src))
    torch.save(out, path)


def letterbox_batch(files, size, device):
    import cv2
    from ultralytics.data.augment import LetterBox

    lb = LetterBox((size, size), auto=False)
    ims = [lb(image=cv2.imread(f))[..., ::-1].transpose(2, 0, 1).copy() for f in files]
    return torch.from_numpy(__import__("numpy").stack(ims)).to(device).float() / 255


@torch.no_grad()
def route(model, files, size, device, bs):
    """The checkpoint's own routing on a folder of images: pooled input of the first banked conv -> router -> top-k."""
    b0 = bank_modules(model)[0]
    st = b0.state
    out = {}
    for s in range(0, len(files), bs):
        x = letterbox_batch(files[s : s + bs], size, device)
        model(x)
        idx = st.logits.topk(st.top_k, 1).indices.cpu().tolist()
        for f, i in zip(files[s : s + bs], idx):
            out[int(os.path.splitext(os.path.basename(f))[0])] = sorted(i)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--pairs", action="store_true", help="also the hard pair branches (six for E=4)")
    ap.add_argument("--merged", action="store_true", help="also the uniform average of all experts")
    ap.add_argument("--route-dir", default=None, help="image folder (val2017) for the checkpoint's own routing")
    ap.add_argument("--route-limit", type=int, default=0)
    ap.add_argument("--check-images", default="/data/datasets/coco/images/val2017", help="folder; first 2 images used")
    ap.add_argument("--perturb", type=float, default=0.0, help="TEST ONLY: perturb experts so branches differ")
    ap.add_argument("--test-top1", action="store_true", help="TEST ONLY: set top_k=1 (as upcycle_fixed.py does) to test the natural-route check")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, f"check_{a.tag}.log"), "w")

    def P(*s):
        print(*s, flush=True)
        print(*s, file=log, flush=True)

    t0 = time.time()
    ck, model = load_model(a.ckpt)
    banks = bank_modules(model)
    st = banks[0].state
    E = st.num_experts
    P(f"ckpt {a.ckpt} epoch {ck.get('epoch')} banks {len(banks)} experts {E} top_k {st.top_k} hard {st.hard} "
      f"router hidden {'0' if isinstance(banks[0].router.fc1, nn.Identity) else banks[0].router.fc1.out_features} "
      f"fixed {getattr(banks[0].router, 'fixed', None)} yaml {model.yaml.get('weight_bank')}")
    if a.perturb > 0:  # smoke test only: make the experts differ, so that a wrong branch would be detected
        g = torch.Generator().manual_seed(0)
        with torch.no_grad():
            for b in banks:
                for k in b.experts:
                    k.add_(a.perturb * k.abs().mean() * torch.randn(k.shape, generator=g))
        st._flat = None
        P(f"TEST ONLY: experts perturbed by {a.perturb} x mean|w|")
    if a.test_top1:
        st.top_k = 1
        banks[0].router.top_k = 1
        P("TEST ONLY: top_k set to 1")
    spread = max(float((torch.stack(tuple(b.experts)) - torch.stack(tuple(b.experts)).mean(0)).abs().max()) for b in banks)
    P(f"max |expert - bank mean| over all banks: {spread:.3e} (0 means the branches are identical)")

    sels = [(i,) for i in range(E)]
    names = [f"{a.tag}_e{i}" for i in range(E)]
    if a.pairs:
        for p in itertools.combinations(range(E), 2):
            sels.append(p)
            names.append(f"{a.tag}_pair{p[0]}{p[1]}")
    if a.merged:
        sels.append(tuple(range(E)))
        names.append(f"{a.tag}_merged")

    files = sorted(glob.glob(os.path.join(a.check_images, "*.jpg")))[:2] if a.check_images and os.path.isdir(a.check_images) else []
    xs = [torch.rand(1, 3, a.size, a.size, generator=torch.Generator().manual_seed(1))]
    if files:
        xs.append(letterbox_batch(files, a.size, "cpu"))
    model_dev = model.to(a.device)
    spec = open(os.path.join(a.out, f"spec_{a.tag}.txt"), "w")
    worst = 0.0
    ref_out = {}
    for sel, nm in zip(sels, names):
        fm, n = force(model, sel)
        fm = fm.to(a.device).eval()
        undo = set_forced_gates(model_dev, sel)
        errs = []
        for j, x in enumerate(xs):
            x = x.to(a.device)
            yb, yf = outputs(model_dev, x), outputs(fm, x)
            errs.append(float((yb - yf).abs().max() / (yb.abs().max() + 1e-9)))
            ref_out[(nm, j)] = yf.cpu()
        undo()
        worst = max(worst, max(errs))
        path = os.path.join(a.out, nm + ".pt")
        save(ck, fm.cpu(), path, sel, a.ckpt)
        spec.write(f"{nm} {path} {a.size} o2o\n")
        P(f"{nm}: experts {list(sel)}, {n} convs replaced, banked-forced vs plain rel. max err {['%.2e' % e for e in errs]}, saved {os.path.getsize(path) / 1e6:.1f} MB")
    spec.close()
    # branches must differ from each other when the experts differ
    if len(sels) > 1:
        d = float((ref_out[(names[0], 0)] - ref_out[(names[1], 0)]).abs().max())
        P(f"head-input difference between {names[0]} and {names[1]} on the random input (must be > 0 when experts differ): {d:.3e}")
    # natural routing must coincide with the forced branch of the selected expert (hard top-1 only)
    if st.top_k == 1 and files:
        with torch.no_grad():
            x = xs[1].to(a.device)
            ynat = outputs(model_dev, x)
            sel_nat = st.logits.argmax(1).tolist()
        for b, e in enumerate(sel_nat):
            err = float((ynat[b] - ref_out[(f"{a.tag}_e{e}", 1)][b].to(ynat.device)).abs().max() / (ynat[b].abs().max() + 1e-9))
            P(f"natural route of check image {b}: expert {e}, routed vs forced rel. max err {err:.2e}")
            worst = max(worst, err)
    # reload one saved file through the public loader, as dump_batch.sh will
    from ultralytics import YOLO

    y = YOLO(os.path.join(a.out, names[0] + ".pt"))
    rel = float((outputs(y.model.float().eval().cpu(), xs[0]) - ref_out[(names[0], 0)]).abs().max() / (ref_out[(names[0], 0)].abs().max() + 1e-9))
    P(f"reload {names[0]}.pt via YOLO(): banks {len(bank_modules(y.model))}, rel. max err vs in-memory (fp16 save) {rel:.2e}")
    worst_reload = rel
    if a.route_dir:
        rf = sorted(glob.glob(os.path.join(a.route_dir, "*.jpg")))
        rf = rf[: a.route_limit] if a.route_limit else rf
        r = route(model_dev, rf, a.size, a.device, a.batch)
        json.dump(r, open(os.path.join(a.out, f"route_{a.tag}.json"), "w"))
        cnt = {}
        for v in r.values():
            cnt[tuple(v)] = cnt.get(tuple(v), 0) + 1
        P(f"routing on {len(r)} images: shares {{{', '.join(f'{k}: {v / len(r):.3f}' for k, v in sorted(cnt.items()))}}}")
    ok = worst < 1e-4 and worst_reload < 2e-2
    P(f"{'PASS' if ok else 'FAIL'}: worst banked-vs-forced rel. max err {worst:.2e} (fp32, tol 1e-4); reload after fp16 save {worst_reload:.2e} (tol 2e-2); {time.time() - t0:.0f} s")
