"""Agent 2 round 2: CPU forward of a YOLO26 checkpoint on val2017 images, one thread, multi-label end2end top-300, conf 0.001.
Variants from ONE pass of model M:  full | noP3 (P3 anchors masked before top-k = 'skip P3 head', exact on public weights) |
const17 (layer 17 output replaced by a per-channel constant, P3 anchors masked: agent 3's branch B, untrained).
Usage: fwd.py WEIGHTS OUTPREFIX START STOP [variants] [--const constfile]"""
import os, sys, json, time, numpy as np, torch, cv2
torch.set_num_threads(1)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
W, OUT, a, b = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
variants = sys.argv[5].split(',') if len(sys.argv) > 5 and not sys.argv[5].startswith('--') else ['full']
constf = sys.argv[sys.argv.index('--const') + 1] if '--const' in sys.argv else None
IMG = '/data/datasets/coco/val2017'
ids = sorted(int(f[:-4]) for f in os.listdir(IMG) if f.endswith('.jpg'))
sel = np.random.RandomState(0).permutation(len(ids))  # fixed random order; chunks [a,b)
ids = [ids[i] for i in sel[a:b]]
coco91 = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
y = YOLO(W); net = y.model.eval().float(); det = net.model[-1]
seq = net.model
const = torch.load(constf) if constf else None
lb = LetterBox((640, 640), auto=False, scaleup=False)
feats17 = []
def run_layers(x, start, cache):
    for m in seq[start:]:
        if m.f != -1:
            x = cache[m.f] if isinstance(m.f, int) else [x if j == -1 else cache[j] for j in m.f]
        x = m(x); cache[m.i] = x
    return x
def topk_out(dec, mask_p3):
    d = dec.clone()
    if mask_p3: d[:, 4:, :6400] = 0
    return det.postprocess(d.permute(0, 2, 1))[0]
res = {v: [] for v in variants}; t0 = time.time()
with torch.no_grad():
    for n, iid in enumerate(ids):
        im0 = cv2.imread(f'{IMG}/{iid:012d}.jpg'); h0, w0 = im0.shape[:2]
        im = lb(image=im0); x = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255
        r = min(640 / h0, 640 / w0, 1.0); pw, ph = (640 - round(w0 * r)) / 2, (640 - round(h0 * r)) / 2
        cache = {}
        out = run_layers(x, 0, cache)
        preds = out[1]['one2one']
        dec = det._inference(preds)
        outs = {}
        if 'full' in variants: outs['full'] = topk_out(dec, False)
        if 'noP3' in variants: outs['noP3'] = topk_out(dec, True)
        if 'feat17' in variants: feats17.append(cache[17].mean((2, 3))[0])
        if 'const17' in variants:
            c2 = dict(cache); c2[17] = const.view(1, -1, 1, 1).expand_as(cache[17])
            xx = c2[17]
            for m in seq[18:]:
                if m.f != -1: xx = c2[m.f] if isinstance(m.f, int) else [xx if j == -1 else c2[j] for j in m.f]
                xx = m(xx); c2[m.i] = xx
            outs['const17'] = topk_out(det._inference(xx[1]['one2one']), True)
        for v, o in outs.items():
            o = o[o[:, 4] >= 0.001].numpy()
            for x1, y1, x2, y2, s, c in o:
                x1 = (x1 - pw) / r; x2 = (x2 - pw) / r; y1 = (y1 - ph) / r; y2 = (y2 - ph) / r
                x1, x2 = np.clip([x1, x2], 0, w0); y1, y2 = np.clip([y1, y2], 0, h0)
                res[v].append({'image_id': iid, 'category_id': coco91[int(c)], 'bbox': [round(float(x1), 3), round(float(y1), 3), round(float(x2 - x1), 3), round(float(y2 - y1), 3)], 'score': round(float(s), 5)})
        if n % 25 == 0: print(n, iid, '%.2fs/img' % ((time.time() - t0) / (n + 1)), flush=True)
for v in variants:
    if v == 'feat17': continue
    json.dump(res[v], open(f'{OUT}_{v}_{a}_{b}.json', 'w'))
if feats17: torch.save(torch.stack(feats17).mean(0), f'{OUT}_const17.pt')
json.dump(ids, open(f'{OUT}_ids_{a}_{b}.json', 'w'))
print('done', len(ids), time.time() - t0, flush=True)
