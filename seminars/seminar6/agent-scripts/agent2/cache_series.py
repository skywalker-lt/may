"""CPU series (one thread): YOLO26-L at 640 on a 500-image val2017 subset, frame t = image translated by d px (grey fill),
with the deep backbone stage cached from frame t-1 (d=0): c10 = layer 10 (P5 after SPPF+C2PSA) cached; c610 = layers 6 and 10 cached.
Writes COCO-format dets per condition for shard k of K."""
import sys, os, json, time, numpy as np, cv2, torch; sys.path.insert(0, "/data/YOLO-Master"); torch.set_num_threads(1)
from ultralytics import YOLO
k, K = int(sys.argv[1]), int(sys.argv[2]); N = 400; SZ = 640; DS = [8, 16]
ids = sorted(json.load(open("/data/tmp/ds-yolo/seminar6/inputs/dumps/instances_val2017.json"))["images"], key=lambda x: x["id"])
rng = np.random.default_rng(0); sel = [ids[i] for i in sorted(rng.choice(len(ids), N, replace=False))][k::K]
m = YOLO("/data/yolo-quant-work/weights/yolo26l.pt").model.eval().float()
cache = {}; mode = {"use": None}
def mk(i):
    def hook(mod, inp, out):
        if mode["use"] is None: cache[i] = out.clone(); return None
        if i in mode["use"]: return cache[i]
    return hook
for i in (6, 10): m.model[i].register_forward_hook(mk(i))
def letterbox(im):
    h, w = im.shape[:2]; r = SZ / max(h, w); nh, nw = round(h * r), round(w * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR); out = np.full((SZ, SZ, 3), 114, np.uint8)
    top, left = (SZ - nh) // 2, (SZ - nw) // 2; out[top:top + nh, left:left + nw] = im
    return out, r, left, top
def shift(im, d):
    out = np.full_like(im, 114); h, w = im.shape[:2]
    out[d:, d:] = im[:h - d, :w - d] if d > 0 else im; return out
def run(im, use):
    x, r, left, top = letterbox(im); mode["use"] = use
    t = torch.from_numpy(x[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    with torch.no_grad(): y = m(t)[0][0].numpy()
    out = []
    for x1, y1, x2, y2, s, c in y:
        out.append(dict(bbox=[round(float((x1 - left) / r), 3), round(float((y1 - top) / r), 3), round(float((x2 - x1) / r), 3), round(float((y2 - y1) / r), 3)], score=round(float(s), 5), category_id=int(c)))
    return out
res = {"full0": []}
for d in DS: res.update({f"full{d}": [], f"c10_{d}": []})
res["c610_8"] = []
t0 = time.time()
for n, info in enumerate(sel):
    im = cv2.imread(f"/data/datasets/coco/images/val2017/{info['file_name']}"); cache.clear()
    for cond, (d, use) in {"full0": (0, None), **{f"full{d}": (d, ()) for d in DS}, **{f"c10_{d}": (d, (10,)) for d in DS}, **{f"c610_{d}": (d, (6, 10)) for d in DS if d == 8}}.items():
        for x in run(shift(im, d), use): res[cond].append(dict(x, image_id=info["id"]))
    if n % 10 == 0: print(k, n, round(time.time() - t0), flush=True)
json.dump(res, open(f"cache_dets_{k}.json", "w"))
print("done", k, round(time.time() - t0))
