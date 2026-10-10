"""Dense absorber D3: single-image test-time BN in the first 21 BN layers (layers 0-4): mean/var = (1-a) running + a per-image
(per-channel over HxW, letterboxed input incl. pad), public statistics elsewhere. a in {0.2, 0.5}. No route."""
import time, json, numpy as np, torch, cv2
import torch.nn.functional as F
from common import *
from corrupt import apply
E = json.load(open(f'{OUT}/eval_ids.json')); sub = E['sub']
gt = json.load(open(GT)); fn = {im['id']: im['file_name'] for im in gt['images']}; cocoid = sorted(c['id'] for c in gt['categories'])
net = load_unfused(); bns = bn_layers(net)[:21]; A = {'a': 0.0}
def fwd(self, x):
    a = A['a']
    if a == 0: return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
    mi = x.mean((0, 2, 3)); vi = x.var((0, 2, 3), unbiased=False)
    mu = (1 - a) * self.running_mean + a * mi; var = (1 - a) * (self.running_var + 0) + a * vi + (1 - a) * a * (mi - self.running_mean) ** 2
    return F.batch_norm(x, mu, var, self.weight, self.bias, False, 0.0, self.eps)
import types
for m in bns: m.forward = types.MethodType(fwd, m)
res = {f'{c}__abn{a}': [] for c in ['clean', 'gn3', 'db3', 'ct3', 'br3'] for a in (0.2, 0.5)}; t0 = time.time()
with torch.no_grad():
    for j, iid in enumerate(sub):
        im = cv2.imread(f'{VAL}/{fn[iid]}'); rng = np.random.default_rng(iid)
        for c in ['clean', 'gn3', 'db3', 'ct3', 'br3']:
            lb, geo = letterbox(apply(im, c, rng)); x = to_tensor(lb)
            for a in (0.2, 0.5):
                A['a'] = a; y = net(x); y = y[0] if isinstance(y, (tuple, list)) else y
                res[f'{c}__abn{a}'] += dets_to_coco(y, iid, geo, cocoid)
        if (j + 1) % 10 == 0: json.dump(res, open(f'{OUT}/matrix4_dets.json', 'w')); print(j + 1, 'images', '%.0fs' % (time.time() - t0), flush=True)
json.dump(res, open(f'{OUT}/matrix4_dets.json', 'w')); print('done', '%.0fs' % (time.time() - t0))
