"""Partial BN experts: family statistics only in the first k BN layers (k=2: layers 0-1; k=21: layers 0-4, output stride 8),
public statistics everywhere else. Same 60 images and same corruption draws as s2 (rng seeded by image id).
Corrupted c under own_k2, own_k21; clean under each family's k2 and k21 set (misroute tax) and cleanrecal_k21."""
import time, json, copy, numpy as np, torch, cv2, os, sys
from common import *
from corrupt import apply
E = json.load(open(f'{OUT}/eval_ids.json')); sub = E['sub']
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}; cocoid = sorted(c['id'] for c in gt['categories'])
base = load_unfused(); sets = torch.load(f'{OUT}/bn_sets.pt'); pub = sets['public']
def partial(name, k): return [sets[name][i] if i < k else pub[i] for i in range(len(pub))]
nets = {}
for s in ['gn3', 'db3', 'ct3', 'br3', 'cleanrecal']:
    for k in ([2, 21] if s != 'cleanrecal' else [21]):
        m = copy.deepcopy(base); set_bn_state(m, partial(s, k)); m.eval(); nets[f'{s}_k{k}'] = m.fuse(verbose=False)
cfg = {'clean': list(nets)}
for c in ['gn3', 'db3', 'ct3', 'br3']: cfg[c] = [f'{c}_k2', f'{c}_k21']
res = {f'{c}__{s}': [] for c in cfg for s in cfg[c]}; t0 = time.time()
with torch.no_grad():
    for j, iid in enumerate(sub):
        im = cv2.imread(f'{VAL}/{fn[iid]}'); rng = np.random.default_rng(iid)
        for c in ['clean', 'gn3', 'db3', 'ct3', 'br3']:  # same order as s2, so the noise draws match
            lb, geo = letterbox(apply(im, c, rng)); x = to_tensor(lb)
            for s in cfg[c]:
                y = nets[s](x); y = y[0] if isinstance(y, (tuple, list)) else y
                res[f'{c}__{s}'] += dets_to_coco(y, iid, geo, cocoid)
        if (j + 1) % 10 == 0: json.dump(res, open(f'{OUT}/matrix2_dets.json', 'w')); print(j + 1, 'images', '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open(f'{OUT}/matrix2_dets.json', 'w')); print('done', '%.0fs' % (time.time() - t0))
