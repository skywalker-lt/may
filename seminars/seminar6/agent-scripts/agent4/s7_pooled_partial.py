"""The dense twin for the partial construct: pooled statistics (equal mix of clean, gn3, db3, ct3, br3) in the first 21 BN
layers, public elsewhere; one static model, no route. Also 'gnct_k21' (statistics from the mix that the route actually uses:
the pooled run is the only multi-condition estimate on disk, so this is approximated by averaging the gn3 and ct3 k21 moments
with the public ones, equal weights, which is the moment-matched pooled set of {clean, gn3, ct3})."""
import time, json, copy, numpy as np, torch, cv2
from common import *
from corrupt import apply
E = json.load(open(f'{OUT}/eval_ids.json')); sub = E['sub']
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}; cocoid = sorted(c['id'] for c in gt['categories'])
base = load_unfused(); sets = torch.load(f'{OUT}/bn_sets.pt'); pub = sets['public']
def mix(names, k):
    out = []
    for i in range(len(pub)):
        if i >= k: out.append(pub[i]); continue
        mus = [sets[s][i][0] for s in names]; vs = [sets[s][i][1] for s in names]
        mu = sum(mus) / len(mus); v = sum(vv + (m - mu) ** 2 for vv, m in zip(vs, mus)) / len(mus); out.append((mu, v))
    return out
cand = {'pooled_k21': [sets['pooled'][i] if i < 21 else pub[i] for i in range(len(pub))], 'gnct_k21': mix(['public', 'gn3', 'ct3'], 21)}
nets = {}
for s, st in cand.items():
    m = copy.deepcopy(base); set_bn_state(m, st); m.eval(); nets[s] = m.fuse(verbose=False)
res = {f'{c}__{s}': [] for c in ['clean', 'gn3', 'db3', 'ct3', 'br3'] for s in nets}; t0 = time.time()
with torch.no_grad():
    for j, iid in enumerate(sub):
        im = cv2.imread(f'{VAL}/{fn[iid]}'); rng = np.random.default_rng(iid)
        for c in ['clean', 'gn3', 'db3', 'ct3', 'br3']:
            lb, geo = letterbox(apply(im, c, rng)); x = to_tensor(lb)
            for s in nets:
                y = nets[s](x); y = y[0] if isinstance(y, (tuple, list)) else y
                res[f'{c}__{s}'] += dets_to_coco(y, iid, geo, cocoid)
        if (j + 1) % 10 == 0: json.dump(res, open(f'{OUT}/matrix3_dets.json', 'w')); print(j + 1, 'images', '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open(f'{OUT}/matrix3_dets.json', 'w')); print('done', '%.0fs' % (time.time() - t0))
