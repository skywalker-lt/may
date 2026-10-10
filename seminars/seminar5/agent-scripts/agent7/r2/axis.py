"""Model axis vs resolution axis at dense M's T4 latency (5.36 ms unset incl. 0.18 ms router): the exchange {M@512, M@640, L@640}
against the one-weight-set menu {M@512, M@640, M@768} and the two-branch {M@512, L@640}; thumbnail count rule, monotone
(sparsest -> cheapest), share nulls (3 draws)."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from headroom import I, cnt_pred_all, cnt_pred_small, n_all, n_small
from mixlib import mix_ap
F = lambda L: 0.5261 + 0.0102 * (L - 5.36); R = 0.18
def rule(menu, cost, q, score):
    """q = share of the cheapest branch; the most expensive branch's share solves the budget at 5.36 ms."""
    c0, c1, c2 = cost; qe = (q * (c1 - c0) - R) / (c2 - c1) if c2 > c1 else 0
    o = np.argsort(score + 1e-9 * np.arange(I)); a = np.ones(I, int); a[o[:int(q * I)]] = 0; a[o[I - int(qe * I):]] = 2
    return a
def show(menu, cost, a, tag):
    ap = mix_ap(menu, a); L = np.array(cost)[a].mean() + R; nl = np.mean([mix_ap(menu, np.random.RandomState(300 + d).permutation(a)) for d in range(3)])
    print(f"{tag:58s} shares {np.round(np.bincount(a, minlength=3)/I,3)} avg {L:.2f} AP {ap:.4f} null {nl:.4f} AP-bar {ap-F(L)-0.003:+.4f} AP-null {ap-nl:+.4f}", flush=True)
s = cnt_pred_all + 1e-6 * cnt_pred_small
X = ["dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco"]; cX = [3.78, 5.36, 6.89]
Rm = ["dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dumpml_yolo26m_768_coco"]; cR = [3.78, 5.36, 6.97]
for q in (0.33, 0.5):
    show(X, cX, rule(X, cX, q, s), f"exchange M512/M640/L640 q={q}")
    show(Rm, cR, rule(Rm, cR, q, s), f"one weight set M512/M640/M768 q={q}")
    ss = cnt_pred_small + 1e-6 * cnt_pred_all
    show(Rm, cR, rule(Rm, cR, q, ss), f"one weight set, small-count score q={q}")
# two branches only: sparsest q -> M@512, rest -> L@640; q solves the budget: q*3.78+(1-q)*6.89+0.18 = 5.36
q2 = (6.89 + R - 5.36) / (6.89 - 3.78); a = np.where(np.argsort(np.argsort(s)) < int(q2 * I), 0, 2)
show(X, cX, a, f"two-branch M512/L640 q={q2:.3f}")
