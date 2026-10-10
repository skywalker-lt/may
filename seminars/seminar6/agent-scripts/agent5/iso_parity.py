"""Round-2 receipt R0b (CPU, one thread): the ISO-296 policy's own shapes against the square letterbox at the same per-image
long side, on the same 150 images (seed 0) as rect_parity.py.  Modes: square at s(image); ISO tight shapes; ISO with a
16-px short-side margin (short = ceil(nh/32 + 0.5) * 32); and rect long-side 640 with the short-side margin, scored against
the saved square-640 detections of rect_parity_640.json."""
import sys, os, json, time, numpy as np, cv2, torch
sys.path.insert(0, '/data/YOLO-Master'); torch.set_num_threads(1)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import io, contextlib
W = '/data/yolo-quant-work/weights/yolo26l.pt'; VAL = '/data/datasets/coco/images/val2017'
GT = '/data/tmp/ds-yolo/seminar6/inputs/dumps/instances_val2017.json'; OUT = '/data/tmp/ds-yolo/seminar6/work/agent5'
N = 150; CAP = 296; R = [448, 512, 544, 576, 640]
gt = COCO(GT); ids = sorted(gt.getImgIds()); rng = np.random.RandomState(0); sub = sorted(rng.choice(ids, N, replace=False).tolist())
cocoid = sorted(gt.getCatIds()); info = {im['id']: (im['width'], im['height']) for im in gt.loadImgs(sub)}
ck = torch.load(W, map_location='cpu', weights_only=False); net = ck['model'].float().eval()
for p in net.parameters(): p.requires_grad_(False)

def shape(s, w, h, pad_short):
    lo, hi = min(w, h), max(w, h); nh = s * lo / hi
    short = int(np.ceil(nh / 32 + pad_short) * 32)
    return (s, short) if w >= h else (short, s)   # (W, H)
def iso_s(w, h):
    ok = [s for s in R if np.prod(shape(s, w, h, 0)) / 1000 <= CAP]; return max(ok) if ok else 448
def prep(im, long_side, WH):
    h, w = im.shape[:2]; r = long_side / max(h, w); nh, nw = round(h * r), round(w * r)
    imr = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else im
    Wd, H = WH; top = (H - nh) // 2; left = (Wd - nw) // 2
    out = np.full((H, Wd, 3), 114, np.uint8); out[top:top + nh, left:left + nw] = imr
    return torch.from_numpy(np.ascontiguousarray(out[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255, (r, left, top)
def run(iid, long_side, WH):
    im = cv2.imread(f'{VAL}/{iid:012d}.jpg'); t, (r, pw, ph) = prep(im, long_side, WH)
    with torch.no_grad(): y = net(t)
    y = y[0] if isinstance(y, (list, tuple)) else y
    out = []
    for x1, y1, x2, y2, s, c in y[0].numpy():
        if s < 0.001: continue
        X1 = (x1 - pw) / r; Y1 = (y1 - ph) / r; X2 = (x2 - pw) / r; Y2 = (y2 - ph) / r
        out.append({'image_id': int(iid), 'category_id': cocoid[int(c)], 'bbox': [float(X1), float(Y1), float(X2 - X1), float(Y2 - Y1)], 'score': float(s)})
    return out
def score(dets, img_ids):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(gt, gt.loadRes(dets), 'bbox'); E.params.imgIds = img_ids; E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0]
def boot(a, b, draws=10):
    out = []
    for _ in range(draws):
        bs = sorted(set(rng.choice(sub, N, replace=True).tolist())); S = set(bs)
        out.append(score([d for d in a if d['image_id'] in S], bs) - score([d for d in b if d['image_id'] in S], bs))
    return np.mean(out), np.std(out)

t0 = time.time(); dets = {'square_s': [], 'iso_tight': [], 'iso_margin': [], 'rect640_margin': []}; kp = {k: [] for k in dets}
for k, iid in enumerate(sub):
    w, h = info[iid]; s = iso_s(w, h)
    for mode, ls, WH in (('square_s', s, (s, s)), ('iso_tight', s, shape(s, w, h, 0)), ('iso_margin', s, shape(s, w, h, 0.5)), ('rect640_margin', 640, shape(640, w, h, 0.5))):
        dets[mode] += run(iid, ls, WH); kp[mode].append(WH[0] * WH[1] / 1000)
    if k % 25 == 24: print(f'  {k+1}/{N} images, {time.time()-t0:.0f}s', flush=True)
json.dump(dets, open(f'{OUT}/iso_parity_dets.json', 'w'))
sq640 = json.load(open(f'{OUT}/rect_parity_640.json'))['square']
ap = {m: score(dets[m], sub) for m in dets}; ap['square640'] = score(sq640, sub)
print('mean kpix per mode:', {m: round(float(np.mean(v)), 1) for m, v in kp.items()}, 'square640 409.6')
print(f'AP on {N} images: ' + '  '.join(f'{m} {v:.4f}' for m, v in ap.items()), flush=True)
for a, b in (('iso_tight', 'square_s'), ('iso_margin', 'square_s'), ('rect640_margin', 'square640'), ('iso_margin', 'iso_tight')):
    m, sd = boot(dets[a], dets[b] if b != 'square640' else sq640)
    print(f'{a} - {b}: {ap[a]-ap[b]:+.4f} (bootstrap mean {m:+.4f}, sd {sd:.4f})', flush=True)
print('done', flush=True)
