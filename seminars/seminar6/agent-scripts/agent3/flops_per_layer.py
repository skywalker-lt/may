import sys, os
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0, "/data/YOLO-Master")
import torch
torch.set_num_threads(1)
from ultralytics.nn.tasks import load_checkpoint as attempt_load_one_weight
import thop, json
res = {}
for name in sys.argv[1:]:
    w = f"/data/yolo-quant-work/weights/{name}.pt"
    model, _ = attempt_load_one_weight(w, device="cpu", inplace=True, fuse=False)
    model.eval()
    layers = list(model.model)
    per = []
    x = torch.zeros(1,3,640,640)
    # run forward layer by layer, counting MACs of each module with thop
    y = []
    tot = 0.0
    with torch.no_grad():
        for m in layers:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            inp = x
            macs, params = thop.profile(m, inputs=(inp,), verbose=False)
            # thop counts MACs as 'macs'
            x = m(inp)
            y.append(x if m.i in model.save else None)
            shape = tuple(x.shape) if torch.is_tensor(x) else [tuple(t.shape) for t in x if torch.is_tensor(t)]
            per.append(dict(i=m.i, type=m.type.split('.')[-1], f=m.f, gmac=macs/1e9, mparams=params/1e6, out=str(shape)))
            tot += macs/1e9
    res[name] = dict(total_gmac=tot, layers=per)
    print(name, "total GMAC", round(tot,3))
    for p in per: print(f"  {p['i']:3d} {p['type']:14s} f={str(p['f']):10s} {p['gmac']:7.3f} GMAC  {p['out']}")
json.dump(res, open("flops_per_layer.json","w"), indent=1)
