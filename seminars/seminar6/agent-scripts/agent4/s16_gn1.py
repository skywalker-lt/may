"""Severity-keyed noise set: BN stem statistics estimated on train2017 at Gaussian noise severity 1 (64 images, batch 8),
then gn1 and clean on the 120 images under gn1_k21 (and gn5 under gn5_k21 from a severity-5 set), paired against public."""
import time, json, copy, os, numpy as np, torch, cv2, io, contextlib
from common import *
from corrupt import apply
sets = torch.load(f'{OUT}/bn_sets.pt'); pub = sets['public']
net = load_unfused(); bns = bn_layers(net)
imgs = sorted(os.listdir(TR)); sel = np.random.RandomState(1).choice(len(imgs), 2000, replace=False); t0 = time.time()
for name, ptr0 in [('gn1', 600), ('gn5', 700)]:
    if name in sets: continue
    set_bn_state(net, pub)
    for m in bns: m.reset_running_stats(); m.momentum = None; m.train()
    rng = np.random.default_rng(11)
    with torch.no_grad():
        for k in range(0, 64, 8):
            net(torch.cat([to_tensor(letterbox(apply(cv2.imread(f'{TR}/{imgs[sel[ptr0 + j]]}'), name, rng))[0]) for j in range(k, k + 8)]))
    for m in bns: m.eval()
    sets[name] = get_bn_state(net); torch.save(sets, f'{OUT}/bn_sets.pt'); print(name, 'set estimated %.0fs' % (time.time() - t0), flush=True)
E = json.load(open(f'{OUT}/eval_ids.json')); sub = E['sub400'][:120]
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}; cocoid = sorted(c['id'] for c in gt['categories'])
def partial(name, k): return [sets[name][i] if i < k else pub[i] for i in range(len(pub))]
nets = {}
for nm in ['gn1', 'gn5']:
    m = copy.deepcopy(load_unfused()); set_bn_state(m, partial(nm, 21)); m.eval(); nets[f'{nm}_k21'] = m.fuse(verbose=False)
cfg = {'gn1': ['gn1_k21'], 'gn5': ['gn5_k21'], 'clean': ['gn1_k21']}
res = {f'{c}__{s}': [] for c in cfg for s in cfg[c]}
with torch.no_grad():
    for j, iid in enumerate(sub):
        im = cv2.imread(f'{VAL}/{fn[iid]}')
        for c in cfg:
            seed = iid + {'1': 100, '5': 500}.get(c[-1], 0) if c != 'clean' else iid
            lb, geo = letterbox(apply(im, c, np.random.default_rng(seed))); x = to_tensor(lb)
            for s in cfg[c]:
                y = nets[s](x); y = y[0] if isinstance(y, (tuple, list)) else y; res[f'{c}__{s}'] += dets_to_coco(y, iid, geo, cocoid)
        if (j + 1) % 20 == 0: print(j + 1, 'images %.0fs' % (time.time() - t0), flush=True)
json.dump(res, open(f'{OUT}/matrix7_dets.json', 'w'))
R6 = json.load(open(f'{OUT}/matrix6_dets.json'))
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
with contextlib.redirect_stdout(io.StringIO()): G = COCO(GT)
def ap(d):
    with contextlib.redirect_stdout(io.StringIO()):
        Ev = COCOeval(G, G.loadRes(d), 'bbox'); Ev.params.imgIds = sub; Ev.evaluate(); Ev.accumulate(); Ev.summarize()
    return Ev.stats[0]
def ap_sub(d, ids):
    dd = [x for x in d if x['image_id'] in ids]
    return ap(dd) if dd else 0.0
rng = np.random.default_rng(0)
for c in ['gn1', 'gn5', 'clean']:
    own = f'{c}__{cfg[c][0]}'; a1, a0 = ap(res[own]), ap(R6[f'{c}__public']); print('%s: own-severity set %.4f public %.4f diff %+.4f' % (c, a1, a0, a1 - a0), flush=True)
# cheap paired sd: 30 half-splits of the images
ds = []
for _ in range(30):
    ids = set(rng.choice(sub, 60, replace=False).tolist()); ds.append(ap_sub(res['gn1__gn1_k21'], ids) - ap_sub(R6['gn1__public'], ids))
print('gn1 own-set minus public over 30 random halves: mean %+.4f sd %.4f' % (np.mean(ds), np.std(ds)))
print('done %.0fs' % (time.time() - t0))
