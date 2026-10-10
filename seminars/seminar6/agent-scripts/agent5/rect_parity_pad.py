"""Round-2 receipt on CPU (one thread): does a native-aspect (rect) input reproduce the square-letterbox AP?
Public YOLO26-L in PyTorch fp32, 150 random val2017 images (seed 0), long side 640 and 576, two preprocessings each:
square letterbox (the protocol) and rect (short side padded to a stride-32 multiple, centred, fill 114).
Scores: pycocotools on the subset, paired per-image bootstrap of the difference."""
import sys, os, json, time, numpy as np, cv2, torch
sys.path.insert(0, '/data/YOLO-Master'); torch.set_num_threads(1)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import io, contextlib
W = '/data/yolo-quant-work/weights/yolo26l.pt'; VAL = '/data/datasets/coco/images/val2017'
GT = '/data/tmp/ds-yolo/seminar6/inputs/dumps/instances_val2017.json'; OUT = '/data/tmp/ds-yolo/seminar6/work/agent5'
N = int(sys.argv[1]) if len(sys.argv) > 1 else 150; PAD = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
gt = COCO(GT); ids = sorted(gt.getImgIds()); rng = np.random.RandomState(0); sub = sorted(rng.choice(ids, N, replace=False).tolist())
cocoid = sorted(gt.getCatIds())
ck = torch.load(W, map_location='cpu', weights_only=False); net = ck['model'].float().eval()
for p in net.parameters(): p.requires_grad_(False)

def prep(im, long_side, rect):
    h, w = im.shape[:2]; r = long_side / max(h, w); nh, nw = round(h * r), round(w * r)
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else im
    if rect: H, Wd = int(np.ceil(nh / 32 + PAD) * 32), int(np.ceil(nw / 32 + PAD) * 32)
    else: H = Wd = long_side
    top = (H - nh) // 2; left = (Wd - nw) // 2
    out = np.full((H, Wd, 3), 114, np.uint8); out[top:top + nh, left:left + nw] = imr
    t = torch.from_numpy(np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255
    return t, (r, left, top), (H, Wd)

def run(iid, long_side, rect):
    im = cv2.imread(f'{VAL}/{iid:012d}.jpg'); t, (r, pw, ph), shp = prep(im, long_side, rect)
    with torch.no_grad(): y = net(t)
    y = y[0] if isinstance(y, (list, tuple)) else y
    d = y[0].numpy(); out = []
    for x1, y1, x2, y2, s, c in d:
        if s < 0.001: continue
        X1 = (x1 - pw) / r; Y1 = (y1 - ph) / r; X2 = (x2 - pw) / r; Y2 = (y2 - ph) / r
        out.append({'image_id': int(iid), 'category_id': cocoid[int(c)], 'bbox': [float(X1), float(Y1), float(X2 - X1), float(Y2 - Y1)], 'score': float(s)})
    return out, shp

def score(dets, img_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets) if dets else gt.loadRes([{'image_id': img_ids[0], 'category_id': 1, 'bbox': [0, 0, 1, 1], 'score': 0.001}]), 'bbox')
        E.params.imgIds = img_ids; E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0], E

t0 = time.time()
for long_side in (640,):
    dets = {'square': [], 'rect': []}; shapes = set()
    for k, iid in enumerate(sub):
        for mode in ('square', 'rect'):
            d, shp = run(iid, long_side, mode == 'rect'); dets[mode] += d
            if mode == 'rect': shapes.add(shp)
        if k % 25 == 24: print(f'  long {long_side}: {k+1}/{N} images, {time.time()-t0:.0f}s', flush=True)
    json.dump(dets, open(f'{OUT}/rect_parity_pad_{long_side}.json', 'w'))
    ap = {m: score(dets[m], sub)[0] for m in dets}
    # paired bootstrap over images: per-image AP is not additive, so resample image subsets and re-score both arms
    diffs = []
    for b in range(10):
        bs = sorted(set(rng.choice(sub, N, replace=True).tolist()))
        diffs.append(score([d for d in dets['rect'] if d['image_id'] in set(bs)], bs)[0] - score([d for d in dets['square'] if d['image_id'] in set(bs)], bs)[0])
    print(f'long side {long_side}, {N} images: square {ap["square"]:.4f}  rect {ap["rect"]:.4f}  rect-square {ap["rect"]-ap["square"]:+.4f} '
          f'(bootstrap over image subsets, 10 draws: mean {np.mean(diffs):+.4f}, sd {np.std(diffs):.4f}); rect shapes used: {len(shapes)}  [{time.time()-t0:.0f}s]', flush=True)
print('done', flush=True)
