"""Cross matrix, trimmed for a loaded CPU: N val2017 images (seed-0 subset). Clean under every set (public, cleanrecal,
pooled, gn3, db3, ct3, br3: the misroute tax on clean images); each corruption at severity 3 under public, pooled and its
own set. BN folded per set (one fused copy per set). One COCO json of all configs, written every 5 images (resumable)."""
import time, json, copy, numpy as np, torch, cv2, os, sys
from common import *
from corrupt import apply
N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
gt = json.load(open(GT)); ids = sorted(im['id'] for im in gt['images']); fn = {im['id']: im['file_name'] for im in gt['images']}
cocoid = sorted(c['id'] for c in gt['categories'])
rng0 = np.random.default_rng(0); sub400 = [ids[i] for i in rng0.choice(len(ids), 400, replace=False)]; sub = sub400[:N]
json.dump({'sub400': sub400, 'sub': sub}, open(f'{OUT}/eval_ids.json', 'w'))
base = load_unfused(); sets = torch.load(f'{OUT}/bn_sets.pt')
nets = {}
for s in ['public', 'cleanrecal', 'pooled', 'gn3', 'db3', 'ct3', 'br3']:
    m = copy.deepcopy(base); set_bn_state(m, sets[s]); m.eval(); nets[s] = m.fuse(verbose=False) if 'verbose' in m.fuse.__code__.co_varnames else m.fuse()
cfg = {'clean': ['public', 'cleanrecal', 'pooled', 'gn3', 'db3', 'ct3', 'br3']}
for c in ['gn3', 'db3', 'ct3', 'br3']: cfg[c] = ['public', 'pooled', c]
path = f'{OUT}/matrix_dets.json'
res = json.load(open(path)) if os.path.exists(path) else {f'{c}__{s}': [] for c in cfg for s in cfg[c]}
done = set(d['image_id'] for d in res['clean__public']) if res['clean__public'] else set()
t0 = time.time(); k = 0
with torch.no_grad():
    for j, iid in enumerate(sub):
        if iid in done: continue
        im = cv2.imread(f'{VAL}/{fn[iid]}'); rng = np.random.default_rng(iid)
        for c in cfg:
            lb, geo = letterbox(apply(im, c, rng)); x = to_tensor(lb)
            for s in cfg[c]:
                y = nets[s](x); y = y[0] if isinstance(y, (tuple, list)) else y
                res[f'{c}__{s}'] += dets_to_coco(y, iid, geo, cocoid)
        k += 1
        if k % 5 == 0 or j == len(sub) - 1:
            json.dump(res, open(path, 'w')); print(j + 1, 'images', '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open(path, 'w')); print('done', '%.0fs' % (time.time() - t0))
