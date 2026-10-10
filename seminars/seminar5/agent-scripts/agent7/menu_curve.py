"""Operating-point menu (whole public YOLO26 models x input scale), thumbnail ridge router: pruned menu, ridge strength,
and the router's learning curve (training images per fold) to bound what a train2017-trained router could add."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from headroom import Xn, prox, I
from budget import solve, F, ROUTER
from mixlib import mix_ap
C2 = {"dump_yolo26s_coco": 2.75, "dumpml_yolo26m_512_coco": 3.78, "dumpml_yolo26m_coco": 5.36, "dump_yolo26l_coco": 6.89, "dumpml_yolo26m_768_coco": 6.97}
menu = list(C2); c = np.array([C2[n] for n in menu]); Y = np.stack([prox(n) for n in menu], 1)
def ridge_pred(X, Y, ntr, lam, k=5, seed=0):
    rng = np.random.RandomState(seed); f = rng.randint(0, k, len(Y)); P = np.zeros_like(Y)
    for j in range(k):
        tr = np.where(f != j)[0]; tr = rng.choice(tr, min(ntr, len(tr)), replace=False); te = f == j
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; A = (X[tr] - mu) / sd; ym = Y[tr].mean(0)
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (Y[tr] - ym)); P[te] = (X[te] - mu) / sd @ W + ym
    return P
def row(P, tag):
    a = solve(P, c, 5.18); ap = mix_ap(menu, a); L = c[a].mean() + ROUTER; sh = np.bincount(a, minlength=len(menu)) / I
    return ap, L, sh
if __name__ == "__main__":
 for lam in (40.0, 400.0, 4000.0):
     ap, L, sh = row(ridge_pred(Xn, Y, 10**9, lam), "")
     print(f"menu5 n320 ridge lam={lam:g} ntr=4000: AP {ap:.4f} avg {L:.2f} shares {np.round(sh,3)} AP-bar {ap-F(L)-0.003:+.4f}", flush=True)
 for ntr in (500, 1000, 2000):
     aps = [row(ridge_pred(Xn, Y, ntr, 400.0, seed=s), "")[0] for s in (0, 1)]
     print(f"menu5 n320 ridge lam=400 ntr={ntr}: AP {np.mean(aps):.4f} (two seeds {np.round(aps,4)})", flush=True)
