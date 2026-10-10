"""Synthetic-motion proxy from still-image dumps: every GT object moves by a random vector of length d (native px, ~640 scale);
'stale' = YOLO26-L@640 boxes from the previous frame (unmoved); 'fresh' = YOLO26-L@448 (or 512) boxes on the moved frame
(dets moved with the object they overlap, IoU>=0.3; background FPs static); 'fusion' = stale (score*gamma) U fresh, class NMS."""
import numpy as np, json, copy, sys, time, torch, torchvision, contextlib, io
from evlib import *
from pycocotools.coco import COCO
D = [0, 4, 8, 16, 32]; GAMMA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.7; NMS_IOU = 0.6
G = gt(); rng = np.random.default_rng(1)
base = json.load(open(PATHS["l640"])); fresh_src = {s: json.load(open(PATHS[s])) for s in ["l448", "l512"]}
def by_img(dets):
    o = {}
    for d in dets: o.setdefault(d["image_id"], []).append(d)
    return o
B = by_img(base); F = {s: by_img(v) for s, v in fresh_src.items()}
def iou_xywh(a, b):
    a = torch.tensor(a, dtype=torch.float32); b = torch.tensor(b, dtype=torch.float32)
    a[:, 2:] += a[:, :2]; b[:, 2:] += b[:, :2]
    return torchvision.ops.box_iou(a, b)
def nms_merge(dets):
    if not dets: return []
    bx = torch.tensor([d["bbox"] for d in dets], dtype=torch.float32); bx[:, 2:] += bx[:, :2]
    sc = torch.tensor([d["score"] for d in dets]); cl = torch.tensor([d["category_id"] for d in dets])
    keep = torchvision.ops.batched_nms(bx, sc, cl, NMS_IOU)[:300]
    return [dets[i] for i in keep.tolist()]
def ap(dets, Gx):
    with contextlib.redirect_stdout(io.StringIO()):
        dt = Gx.loadRes(dets); E = COCOeval(Gx, dt, "bbox"); E.evaluate(); E.accumulate(); E.summarize()
    return round(float(E.stats[0]), 4), round(float(E.stats[3]), 4), round(float(E.stats[4]), 4), round(float(E.stats[5]), 4)
print(f"gamma={GAMMA} nms_iou={NMS_IOU}", flush=True)
for d in D:
    t0 = time.time()
    # displaced GT
    Gd = COCO(); Gd.dataset = copy.deepcopy(G.dataset)
    moves = {}  # image -> (gt boxes orig, vectors)
    for a in Gd.dataset["annotations"]:
        th = rng.random() * 2 * np.pi; v = (d * np.cos(th), d * np.sin(th))
        moves.setdefault(a["image_id"], ([], []))
        moves[a["image_id"]][0].append(list(a["bbox"])); moves[a["image_id"]][1].append(v)
        a["bbox"] = [a["bbox"][0] + v[0], a["bbox"][1] + v[1], a["bbox"][2], a["bbox"][3]]
    with contextlib.redirect_stdout(io.StringIO()): Gd.createIndex()
    stale = base
    res = {"stale": ap(stale, Gd)}
    for s in (["l448", "l512"] if d in (8, 16) else ["l448"]):
        fresh = []; fused = []
        for img, dl in F[s].items():
            dl2 = [dict(x) for x in dl]
            if img in moves and dl2:
                gb, gv = moves[img]; M = iou_xywh([x["bbox"] for x in dl2], gb); mx, ix = M.max(1)
                for k, x in enumerate(dl2):
                    if mx[k] >= 0.3:
                        v = gv[ix[k]]; x["bbox"] = [x["bbox"][0] + v[0], x["bbox"][1] + v[1], x["bbox"][2], x["bbox"][3]]
            fresh += dl2
            st = [dict(x, score=x["score"] * GAMMA) for x in B.get(img, [])]
            fused += nms_merge(st + dl2)
        # images with stale dets but no fresh dets
        for img in set(B) - set(F[s]): fused += [dict(x, score=x["score"] * GAMMA) for x in B[img]]
        if d == 0 or s == "l448" and d == 8: res[f"fresh_{s}"] = ap(fresh, Gd)
        res[f"fusion_{s}"] = ap(fused, Gd)
    print(f"d={d:2d} " + "  ".join(f"{k}: AP={v[0]:.4f} S/M/L={v[1]:.3f}/{v[2]:.3f}/{v[3]:.3f}" for k, v in res.items()), f"[{time.time()-t0:.0f}s]", flush=True)
