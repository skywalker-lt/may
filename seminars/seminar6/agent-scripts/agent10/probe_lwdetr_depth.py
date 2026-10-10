"""Agent 10 round 1: LW-DETR-large (public 60e COCO, EMA) on a val2017 subset. One backbone+encoder+3-layer decoder
pass per image; records the top-300 detections of every exit: d0 = two-stage encoder proposals (no decoder layer),
d1, d2, d3 = decoder heads (deep supervision). Signals per image: top-30 score mass and box IoU between consecutive
exits. ONE CPU thread. Output npz in the same format as probe_rtdetr_depth.py (keys d{d}_K300)."""
import os, sys, time, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/data/LW-DETR")
from main import get_args_parser
from models import build_model
from util.utils import clean_state_dict
from PIL import Image
import torchvision.transforms.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5); ap.add_argument("--stride", type=int, default=20)
ap.add_argument("--out", default="/data/tmp/ds-yolo/seminar6/work/agent10/lw_probe.npz")
a = ap.parse_args()
largs = open("/data/lwdetr-work/large_args.txt").read().split()
largs = [x for x in largs if x not in ("--eval", "--use_ema")]
i = largs.index("--resume"); largs = largs[:i] + largs[i + 2:]
args = get_args_parser().parse_args(largs + ["--device", "cpu"])
model, _, _ = build_model(args)
ck = torch.load("/data/lwdetr-work/ckpts/LWDETR_large_60e_coco.pth", map_location="cpu", weights_only=False)
model.load_state_dict(clean_state_dict(ck["ema_model"] if "ema_model" in ck else ck["model"]), strict=True)
model.eval()
ann = json.load(open("/data/datasets/coco/annotations/instances_val2017.json"))
imgs = sorted(ann["images"], key=lambda x: x["id"])[:: a.stride][: a.n]
print("images", len(imgs), "ema" if "ema_model" in ck else "model", flush=True)
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
def iou(a_, b):
    ax1, ay1, ax2, ay2 = a_[:,0]-a_[:,2]/2, a_[:,1]-a_[:,3]/2, a_[:,0]+a_[:,2]/2, a_[:,1]+a_[:,3]/2
    bx1, by1, bx2, by2 = b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2
    iw = (np.minimum(ax2,bx2)-np.maximum(ax1,bx1)).clip(0); ih = (np.minimum(ay2,by2)-np.maximum(ay1,by1)).clip(0)
    inter = iw*ih; return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-9)
rec, sig, t0 = {}, [], time.time()
def save(n):
    np.savez_compressed(a.out, **{f"d{d}_K300": np.concatenate(v) for d, v in rec.items()},
                        signals=json.dumps(sig), timings=json.dumps({"backbone": 0, "encoder": 0, "dec300": 1, "dec100": 1}), n=n)
with torch.no_grad():
    for n, im in enumerate(imgs):
        pil = Image.open(f"/data/datasets/coco/val2017/{im['file_name']}").convert("RGB"); W, H = pil.size
        x = F.normalize(F.to_tensor(F.resize(pil, [640, 640])), mean, std)[None]
        out = model(x)
        exits = [(out["enc_outputs"]["pred_logits"], out["enc_outputs"]["pred_boxes"])] + \
                [(o["pred_logits"], o["pred_boxes"]) for o in out["aux_outputs"]] + [(out["pred_logits"], out["pred_boxes"])]
        row = {"id": im["id"]}; prev = None
        for d, (lg, bx) in enumerate(exits):
            s_all = lg[0].sigmoid().numpy(); b = bx[0].numpy(); s = s_all.reshape(-1)
            idx = np.argpartition(-s, 300)[:300]; idx = idx[np.argsort(-s[idx])]
            q, c = idx // s_all.shape[1], idx % s_all.shape[1]; bb = b[q]
            xywh = np.stack([(bb[:,0]-bb[:,2]/2)*W, (bb[:,1]-bb[:,3]/2)*H, bb[:,2]*W, bb[:,3]*H], 1)
            rec.setdefault(d, []).append(np.concatenate([np.full((300,1), im["id"]), c[:,None], s[idx][:,None], xywh], 1).astype(np.float32))
            top = np.argsort(-s_all.max(1))[:30]
            row[f"s{d}"] = float(s_all.max(1)[top].mean()); row[f"n{d}"] = int((s_all.max(1) > 0.3).sum())
            if prev is not None: row[f"iou{d}"] = float(iou(b[top], prev[top]).mean())
            prev = b
        sig.append(row)
        if (n+1) % 25 == 0 or n < 3: print(f"{n+1}/{len(imgs)} {(time.time()-t0)/(n+1):.2f} s/img", flush=True)
        if (n+1) % 50 == 0: save(n+1)
save(len(imgs)); print("done", time.time()-t0)
