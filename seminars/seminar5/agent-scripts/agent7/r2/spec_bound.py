"""Bound on what a routing-specialised cheap member could add: the q=0.50 exchange with the sparse half scored by M@640 instead of
M@512 (a 512 member as good as M@640 on its routed images, at 512 cost), and the same at q=0.55 two-branch."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from headroom import I, cnt_pred_all, cnt_pred_small
from mixlib import mix_ap
X = ["dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco"]
s = cnt_pred_all + 1e-6 * cnt_pred_small; r = np.empty(I, int); r[np.argsort(s + 1e-9 * np.arange(I))] = np.arange(I)
for q, qL in ((0.50, (0.5 * 1.58 - 0.18) / 1.53), (0.55, 0.45)):
    a = np.ones(I, int); a[r < int(q * I)] = 0; a[r >= I - int(qL * I)] = 2
    b = a.copy(); b[a == 0] = 1
    print(f"q={q}: exchange {mix_ap(X, a):.4f}; sparse share scored by M@640 (specialisation bound) {mix_ap(X, b):.4f}; gain bound {mix_ap(X, b)-mix_ap(X, a):+.4f}", flush=True)
