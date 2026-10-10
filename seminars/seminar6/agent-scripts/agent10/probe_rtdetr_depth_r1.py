"""Agent 10 probe: RT-DETRv2-R50 on a val2017 subset, every decoder depth d=1..6 and query count K in {100,300},
captured from ONE backbone+encoder pass per image (decoder run twice). One CPU thread. Output: per-image per-(d,K)
top-300 detections in COCO format (npz), plus per-stage CPU timings."""
import os, sys, time, json, argparse
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, "/data/RT-DETR/rtdetrv2_pytorch")
from src.core import YAMLConfig
from src.data.dataset.coco_dataset import mscoco_label2category
from PIL import Image
import torchvision.transforms.v2.functional as TF

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--stride", type=int, default=5)
ap.add_argument("--out", default="/data/tmp/ds-yolo/seminar6/work/agent10/probe_out.npz")
ap.add_argument("--Ks", default="300,100")
args = ap.parse_args()
Ks = [int(k) for k in args.Ks.split(",")]

cfg = YAMLConfig("/data/RT-DETR/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml",
                 PResNet={"pretrained": False})
model = cfg.model
ck = torch.load("/data/rtdetr-work/ckpts/rtdetrv2_r50vd_6x_coco_ema.pth", map_location="cpu")
model.load_state_dict(ck["ema"]["module"])
model.eval()
dec = model.decoder                 # RTDETRTransformerv2
dec.decoder.training = True         # TransformerDecoder keeps every layer's output (dropout is 0)

ann = json.load(open("/data/datasets/coco/annotations/instances_val2017.json"))
imgs = sorted(ann["images"], key=lambda x: x["id"])[:: args.stride][: args.n]
print("images", len(imgs), flush=True)

label2cat = np.array([mscoco_label2category[i] for i in range(80)])
rec = {}   # (d,K) -> list of arrays [n_det, 6]: image_id, cat, score, x, y, w, h
T = {"backbone": 0.0, "encoder": 0.0, "dec300": 0.0, "dec100": 0.0}
sig_rows = []  # per image signals: for K=300: per-layer mean top-30 score, mean IoU of top-30 boxes between layer d-1 and d
def box_iou_cxcywh(a, b):
    ax1, ay1, ax2, ay2 = a[:,0]-a[:,2]/2, a[:,1]-a[:,3]/2, a[:,0]+a[:,2]/2, a[:,1]+a[:,3]/2
    bx1, by1, bx2, by2 = b[:,0]-b[:,2]/2, b[:,1]-b[:,3]/2, b[:,0]+b[:,2]/2, b[:,1]+b[:,3]/2
    iw = (np.minimum(ax2,bx2)-np.maximum(ax1,bx1)).clip(0); ih = (np.minimum(ay2,by2)-np.maximum(ay1,by1)).clip(0)
    inter = iw*ih; return inter/((ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter+1e-9)

t_all = time.time()
with torch.no_grad():
    for n, im in enumerate(imgs):
        pil = Image.open(f"/data/datasets/coco/val2017/{im['file_name']}").convert("RGB")
        W, H = pil.size
        x = TF.to_dtype(TF.to_image(TF.resize(pil, [640, 640])), torch.float32, scale=True)[None]
        t0 = time.time(); feats = model.backbone(x); t1 = time.time(); feats = model.encoder(feats); t2 = time.time()
        T["backbone"] += t1 - t0; T["encoder"] += t2 - t1
        row = {"id": im["id"]}
        for K in Ks:
            dec.num_queries = K
            t3 = time.time()
            memory, spatial_shapes = dec._get_encoder_input(feats)
            content, ref_unact, _, _ = dec._get_decoder_input(memory, spatial_shapes, None, None)
            bboxes, logits = dec.decoder(content, ref_unact, memory, spatial_shapes, dec.dec_bbox_head,
                                         dec.dec_score_head, dec.query_pos_head, attn_mask=None)
            T[f"dec{K}"] += time.time() - t3
            bboxes = bboxes[:, 0].numpy(); logits = logits[:, 0].numpy()   # (6,K,4), (6,K,80)
            scores_all = 1 / (1 + np.exp(-logits))
            for d in range(6):
                s = scores_all[d].reshape(-1)
                idx = np.argpartition(-s, 300)[:300]; idx = idx[np.argsort(-s[idx])]
                q, c = idx // 80, idx % 80
                b = bboxes[d][q]
                xywh = np.stack([(b[:,0]-b[:,2]/2)*W, (b[:,1]-b[:,3]/2)*H, b[:,2]*W, b[:,3]*H], 1)
                out = np.concatenate([np.full((300,1), im["id"]), label2cat[c][:,None], s[idx][:,None], xywh], 1)
                rec.setdefault((d+1, K), []).append(out.astype(np.float32))
            if K == 300:
                for d in range(6):
                    top = np.argsort(-scores_all[d].max(1))[:30]
                    row[f"s{d+1}"] = float(scores_all[d].max(1)[top].mean())
                    row[f"n{d+1}"] = int((scores_all[d].max(1) > 0.3).sum())
                    if d > 0:
                        row[f"iou{d+1}"] = float(box_iou_cxcywh(bboxes[d][top], bboxes[d-1][top]).mean())
        sig_rows.append(row)
        if (n+1) % 25 == 0 or n < 5:
            print(f"{n+1}/{len(imgs)}  {(time.time()-t_all)/(n+1):.2f} s/img  {T}", flush=True)
        if (n+1) % 50 == 0:
            np.savez_compressed(args.out, **{f"d{d}_K{K}": np.concatenate(v) for (d, K), v in rec.items()},
                                signals=json.dumps(sig_rows), timings=json.dumps(T), n=n+1)

np.savez_compressed(args.out, **{f"d{d}_K{K}": np.concatenate(v) for (d, K), v in rec.items()},
                    signals=json.dumps(sig_rows), timings=json.dumps(T), n=len(imgs))
print("done", time.time() - t_all, T)
