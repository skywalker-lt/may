"""q=0.50 exchange rule: full-val share null (3 draws) and a paired image bootstrap (20 resamples) of rule - dense M."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from headroom import I, cnt_pred_small, cnt_pred_all
from mixlib import mix_ap, mix_ap_index
menu = ["dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco"]
s = cnt_pred_all + 1e-6 * cnt_pred_small; o = np.argsort(s); q = 0.50; qL = (q * 1.58 - 0.18) / 1.53
a = np.ones(I, int); a[o[:int(q * I)]] = 0; a[o[I - int(qL * I):]] = 2
nl = [mix_ap(menu, np.random.RandomState(80 + d).permutation(a)) for d in range(3)]
print(f"q=0.50 rule {mix_ap(menu, a):.4f} null {np.mean(nl):.4f} draws {np.round(nl,4)}", flush=True)
d = []
for b in range(20):
    sel = np.random.RandomState(1000 + b).randint(0, I, I)
    d.append(mix_ap_index(menu, a, sel) - mix_ap_index(menu, np.ones(I, int), sel))
print(f"paired bootstrap rule - dense M: mean {np.mean(d):+.4f} sd {np.std(d, ddof=1):.4f} (20 resamples)", flush=True)
