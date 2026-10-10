"""Robustness envelope over input scale: public YOLO26-M at 448 and 512 (public BN), clean and gn3 (same noise draws as s2,
noise added at native resolution before letterbox), same 60 images. Does a smaller input absorb the noise gain?"""
import time, json, copy, numpy as np, torch, cv2
from common import *
from corrupt import apply
E = json.load(open(f'{OUT}/eval_ids.json')); sub = E['sub']
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}; cocoid = sorted(c['id'] for c in gt['categories'])
net = load_unfused().fuse(verbose=False)
res = {f'{c}__public@{S}': [] for c in ['clean', 'gn3'] for S in (448, 512)}; t0 = time.time()
with torch.no_grad():
    for iid in sub:
        im = cv2.imread(f'{VAL}/{fn[iid]}'); rng = np.random.default_rng(iid)
        for c in ['clean', 'gn3']:
            imc = apply(im, c, rng)
            for S in (448, 512):
                lb, geo = letterbox(imc, S); y = net(to_tensor(lb)); y = y[0] if isinstance(y, (tuple, list)) else y
                res[f'{c}__public@{S}'] += dets_to_coco(y, iid, geo, cocoid)
json.dump(res, open(f'{OUT}/matrix5_dets.json', 'w')); print('done %.0fs' % (time.time() - t0))
import io, contextlib
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
with contextlib.redirect_stdout(io.StringIO()): G = COCO(GT)
for k, v in res.items():
    with contextlib.redirect_stdout(io.StringIO()):
        Ev = COCOeval(G, G.loadRes(v), 'bbox'); Ev.params.imgIds = sub; Ev.evaluate(); Ev.accumulate(); Ev.summarize()
    print(k, '%.4f' % Ev.stats[0])
