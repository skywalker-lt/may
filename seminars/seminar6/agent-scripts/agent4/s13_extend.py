"""Round 2: (i) images 60-119 for the key cells (clean/gn3/ct3 x public/own-stem/pooled-stem), (ii) severities 1 and 5 of
noise and 1 of contrast on all 120 images under public and the own stem set (estimated at severity 3: mismatch test).
Noise draws seeded by image id as before; severity-1/5 draws use seed id+100/id+500."""
import time, json, copy, numpy as np, torch, cv2, os
from common import *
from corrupt import apply
E = json.load(open(f'{OUT}/eval_ids.json')); sub120 = E['sub400'][:120]
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}; cocoid = sorted(c['id'] for c in gt['categories'])
base = load_unfused(); sets = torch.load(f'{OUT}/bn_sets.pt'); pub = sets['public']
def partial(name, k): return [sets[name][i] if i < k else pub[i] for i in range(len(pub))]
nets = {}
for nm, st in [('public', pub), ('gn3_k21', partial('gn3', 21)), ('ct3_k21', partial('ct3', 21)), ('pooled_k21', partial('pooled', 21))]:
    m = copy.deepcopy(base); set_bn_state(m, st); m.eval(); nets[nm] = m.fuse(verbose=False)
def cfg_for(j):
    c = {'gn1': ['public', 'gn3_k21'], 'gn5': ['public', 'gn3_k21'], 'ct1': ['public', 'ct3_k21']}
    if j >= 60: c.update({'clean': ['public', 'gn3_k21', 'ct3_k21', 'pooled_k21'], 'gn3': ['public', 'gn3_k21', 'pooled_k21'], 'ct3': ['public', 'ct3_k21', 'pooled_k21']})
    return c
path = f'{OUT}/matrix6_dets.json'; res = json.load(open(path)) if os.path.exists(path) else {}
done = set(res.get('_done', [])); t0 = time.time()
with torch.no_grad():
    for j, iid in enumerate(sub120):
        if iid in done: continue
        im = cv2.imread(f'{VAL}/{fn[iid]}'); cfg = cfg_for(j)
        for c in cfg:
            seed = iid + {'1': 100, '5': 500}.get(c[-1], 0) if c != 'clean' else iid
            lb, geo = letterbox(apply(im, c, np.random.default_rng(seed))); x = to_tensor(lb)
            for s in cfg[c]:
                y = nets[s](x); y = y[0] if isinstance(y, (tuple, list)) else y
                res.setdefault(f'{c}__{s}', []).extend(dets_to_coco(y, iid, geo, cocoid))
        done.add(iid); res['_done'] = sorted(done)
        if len(done) % 10 == 0: json.dump(res, open(path, 'w')); print(len(done), 'images', '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open(path, 'w')); print('done', '%.0fs' % (time.time() - t0))
