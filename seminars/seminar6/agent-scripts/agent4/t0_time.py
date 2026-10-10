import time, numpy as np, torch, cv2, json
from common import *
from corrupt import apply
net = load_unfused(); print('bn layers', len(bn_layers(net)))
im = cv2.imread(f'{VAL}/000000000139.jpg')
rng = np.random.default_rng(0)
for cond in ['clean', 'gn3', 'db3', 'ct3', 'br3']:
    t = time.time(); imc = apply(im, cond, rng); tc = time.time() - t
    lb, geo = letterbox(imc); x = to_tensor(lb)
    t = time.time()
    with torch.no_grad(): y = net(x)
    y = y[0] if isinstance(y, (tuple, list)) else y
    print(cond, 'corrupt %.3fs' % tc, 'fwd %.2fs' % (time.time() - t), 'out', tuple(y.shape), 'top score %.3f' % y[0, 0, 4].item(), 'n>0.25', int((y[0, :, 4] > 0.25).sum()), flush=True)
    cv2.imwrite(f'{OUT}/sample_{cond}.jpg', imc)
