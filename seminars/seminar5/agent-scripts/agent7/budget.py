"""Budgeted mixture of whole YOLO26 models (no shared stem) on the T4 cost basis (unset-buffer standalone ms).
Policies: proxy oracle (argmax p_j - lam c_j with GT per-image proxy AP), learned (same with out-of-fold ridge predictions of p_j
from the n320 thumbnail feature), share-matched null (random route with the learned policy's shares). Exact pycocotools AP."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from headroom import ridge_oof, Xn, prox, I
from mixlib import mix_ap, load
from scipy.stats import spearmanr
F = lambda L: 0.5261 + 0.0102 * (L - 5.36)
C = {"dump_yolo26n_coco": 1.65, "dump_yolo26s_coco": 2.75, "dumpml_yolo26m_coco": 5.36, "dump_yolo26l_coco": 6.89, "dump_yolo26x_coco": 12.41}
ROUTER = 0.18  # n320 thumbnail stem on the T4 (stem_n_320 0.178-0.179 ms, batchP2.log)
def policy(P, c, lam): return np.argmax(P - lam * c[None, :], axis=1)
def solve(P, c, target):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        lam = (lo + hi) / 2; a = policy(P, c, lam)
        if c[a].mean() > target: lo = lam
        else: hi = lam
    return policy(P, c, hi)
def run(names, targets, tag):
    c = np.array([C[n] for n in names]); Pg = np.stack([prox(n) for n in names], 1); Pl = np.stack([ridge_oof(Xn, prox(n)) for n in names], 1)
    print(f"== {tag}: " + ", ".join(f"{n.split('_')[1]}={load(n)['stats'][0]:.4f}@{C[n]}" for n in names), flush=True)
    for j, n in enumerate(names):
        if j: print(f"   spearman(proxy gain {n.split('_')[1]} - {names[0].split('_')[1]}, learned prediction) = {spearmanr(Pg[:, j]-Pg[:, 0], Pl[:, j]-Pl[:, 0])[0]:+.3f}")
    for t in targets:
        ao, al = solve(Pg, c, t), solve(Pl, c, t)
        sh = np.bincount(al, minlength=len(names)) / I; rng = np.random.RandomState(7)
        nulls = []
        for d in range(2):
            an = rng.permutation(al); nulls.append(mix_ap(names, an))
        o, l = mix_ap(names, ao), mix_ap(names, al); L = c[al].mean() + ROUTER; Lo = c[ao].mean() + ROUTER
        print(f"   budget {t:.2f} ms: oracle(proxy) {o:.4f} (avg {Lo:.2f}, F+0.003 {F(Lo)+0.003:.4f}) | learned {l:.4f} avg {L:.2f} ms incl. router, shares {np.round(sh, 3)} | "
              f"null {np.mean(nulls):.4f} | bar F(avg)+0.003 {F(L)+0.003:.4f} | learned-bar {l-F(L)-0.003:+.4f} learned-null {l-np.mean(nulls):+.4f}", flush=True)
if __name__ == "__main__":
    n, s, m, l, x = list(C)
    run([m, x], [5.18, 6.0], "M/X")
    run([s, m, l], [5.18], "S/M/L")
    run([s, m, x], [5.18], "S/M/X")
    run([n, s, m, l, x], [5.18], "N/S/M/L/X")
