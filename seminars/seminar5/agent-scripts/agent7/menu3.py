"""The construct's core: {M@512, M@640, L@640} whole-model menu ("resolution down, capacity up") and the pruned 5-point menu:
learned thumbnail router, GT-descriptor bound, a two-quantile thumbnail count rule, and share-matched nulls (3 draws)."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from headroom import Xn, prox, I, cnt_pred_small, cnt_pred_all, n_small, n_all
from budget import solve, F, ROUTER
from menu_curve import ridge_pred
from budget2 import Xg
from mixlib import mix_ap
def report(menu, c, a, tag):
    ap = mix_ap(menu, a); L = c[a].mean() + ROUTER; sh = np.bincount(a, minlength=len(menu)) / I
    nl = [mix_ap(menu, np.random.RandomState(50 + d).permutation(a)) for d in range(3)]
    print(f"{tag:48s} AP {ap:.4f} avg {L:.2f} shares {np.round(sh,3)} null {np.mean(nl):.4f} (sd {np.std(nl):.4f}) bar {F(L)+0.003:.4f} AP-bar {ap-F(L)-0.003:+.4f} AP-null {ap-np.mean(nl):+.4f}", flush=True)
    return a
if True:
    pass
M3 = {"dumpml_yolo26m_512_coco": 3.78, "dumpml_yolo26m_coco": 5.36, "dump_yolo26l_coco": 6.89}
menu = list(M3); c = np.array([M3[n] for n in menu]); Y = np.stack([prox(n) for n in menu], 1)
if __name__ == "__main__":
    for lam in (400.0, 4000.0):
        report(menu, c, solve(ridge_pred(Xn, Y, 10**9, lam), c, 5.18), f"menu3 n320 ridge lam={lam:g}")
    report(menu, c, solve(ridge_pred(Xg, Y, 10**9, 40.0), c, 5.18), "menu3 GT descriptors (bound)")
    for q in (0.25, 0.33):  # thumbnail-predicted count: lowest q -> M@512, highest q' -> L, q' from the budget
        s = cnt_pred_all + 1e-6 * cnt_pred_small; o = np.argsort(s); qL = q * (5.36 - 3.78) / (6.89 - 5.36) - 0.18 / (6.89 - 5.36)
        a = np.ones(I, int); a[o[:int(q * I)]] = 0; a[o[I - int(qL * I):]] = 2
        report(menu, c, a, f"menu3 thumbnail count rule q512={q}")
        sg = n_all + 1e-3 * n_small; o = np.argsort(sg); a = np.ones(I, int); a[o[:int(q * I)]] = 0; a[o[I - int(qL * I):]] = 2
        report(menu, c, a, f"menu3 GT count rule q512={q} (bound)")
    M5 = {"dump_yolo26s_coco": 2.75, "dumpml_yolo26m_512_coco": 3.78, "dumpml_yolo26m_coco": 5.36, "dump_yolo26l_coco": 6.89, "dumpml_yolo26m_768_coco": 6.97}
    menu = list(M5); c = np.array([M5[n] for n in menu]); Y = np.stack([prox(n) for n in menu], 1)
    report(menu, c, solve(ridge_pred(Xn, Y, 10**9, 4000.0), c, 5.18), "menu5 n320 ridge lam=4000")
    report(menu, c, solve(ridge_pred(Xg, Y, 10**9, 40.0), c, 5.18), "menu5 GT descriptors (bound)")
