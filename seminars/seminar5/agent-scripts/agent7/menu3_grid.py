"""menu3 thumbnail count rule: full q grid on the unset basis and on the real-input basis (M@512 3.47 in the two-scale engine,
M 5.05, L 6.55 standalone real, batchP3/res2 logs), plus the per-tercile proxy AP of each member (why the rule works)."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from headroom import prox, I, cnt_pred_small, cnt_pred_all, n_all
from menu3 import report, menu, c
from mixlib import mix_ap
s = cnt_pred_all + 1e-6 * cnt_pred_small; o = np.argsort(s)
for basis, (c512, cm, cl) in (("unset", (3.78, 5.36, 6.89)), ("real", (3.47, 5.05, 6.55))):
    for q in ((0.20, 0.30, 0.33, 0.36, 0.40, 0.45, 0.50) if basis == "unset" else (0.33, 0.40)):
        qL = (q * (cm - c512) - 0.18) / (cl - cm); a = np.ones(I, int); a[o[:int(q * I)]] = 0; a[o[I - int(qL * I):]] = 2
        ap = mix_ap(menu, a); L = np.array([c512, cm, cl])[a].mean() + 0.18
        print(f"{basis} q512={q:.2f} qL={qL:.3f}: AP {ap:.4f} avg {L:.3f} ms (dense M {cm}) AP-0.5291 {ap-0.5291:+.4f}", flush=True)
t = np.digitize(s, np.quantile(s, [1/3, 2/3]))
for nm in menu:
    p = prox(nm); print(nm, "proxy AP by predicted-count tercile (low/mid/high):", [round(float(p[t == k].mean()), 4) for k in range(3)])
