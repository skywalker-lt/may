"""BN-statistics experts by recalibration (no gradient): public YOLO26-M, BN running stats re-estimated (cumulative
average, batch 8, BN modules in train mode, everything else eval) on train2017 images under one condition.
Sets: cleanrecal (control), gn3, db3, ct3, br3 (one per corruption), pooled (equal mix of the five conditions)."""
import time, numpy as np, torch, cv2, os
from common import *
from corrupt import apply
net = load_unfused(); bns = bn_layers(net)
public = get_bn_state(net)
imgs = sorted(os.listdir(TR)); rs = np.random.RandomState(1); sel = rs.choice(len(imgs), 2000, replace=False)
NPER = int(os.environ.get('NPER', 64)); B = 8
conds = ['clean', 'gn3', 'db3', 'ct3', 'br3']
sets = {'public': public}
t0 = time.time(); ptr = 0
for name in ['cleanrecal', 'gn3', 'db3', 'ct3', 'br3', 'pooled']:
    rng = np.random.default_rng(hash(name) % 2**32)
    set_bn_state(net, public)
    for m in bns: m.reset_running_stats(); m.momentum = None; m.train()
    n = NPER if name != 'pooled' else NPER * 5 // 4  # 80 for pooled: 16 per condition
    with torch.no_grad():
        for k in range(0, n, B):
            xs = []
            for j in range(k, min(k + B, n)):
                im = cv2.imread(f'{TR}/{imgs[sel[ptr]]}'); ptr += 1
                c = {'cleanrecal': 'clean', 'pooled': conds[(j) % 5]}.get(name, name)
                xs.append(to_tensor(letterbox(apply(im, c, rng))[0]))
            net(torch.cat(xs))
    for m in bns: m.eval()
    sets[name] = get_bn_state(net)
    # summary: mean |shift| of BN means in units of public std, first 5 BN layers and all
    d = [((mu - mu0).abs() / (v0.sqrt() + 1e-6)).mean().item() for (mu, v), (mu0, v0) in zip(sets[name], public)]
    print(name, 'n', n, 'mean|dmu|/sd layers0-4 %.3f all %.3f  var ratio median %.3f' % (np.mean(d[:5]), np.mean(d),
          np.median([(v / v0).median().item() for (mu, v), (mu0, v0) in zip(sets[name], public)])), '%.0fs' % (time.time() - t0), flush=True)
torch.save(sets, f'{OUT}/bn_sets.pt'); print('saved', '%.0fs' % (time.time() - t0))
