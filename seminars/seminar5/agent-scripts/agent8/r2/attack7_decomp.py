"""Round 2, attack on direction 7: decompose exchange routing {M@512, M@640, L@640} at q512=0.50, qL=0.399 into
(b) shrink only (agent 1's RS-2 with agent 7's router), (d) L by count only (seminar-4 ML), (c) shrink by count + random L.
Uses agent 7's evalImgs caches and router read-only. Front F(L)=0.5261+0.0102(L-5.36), unset basis, router 0.18 ms."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"; sys.dont_write_bytecode = True
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7")
from headroom import cnt_pred_all, cnt_pred_small, I
from mixlib import mix_ap
F = lambda L: 0.5261 + 0.0102 * (L - 5.36)
menu = ["dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco"]; c = np.array([3.78, 5.36, 6.89]); R = 0.18
s = cnt_pred_all + 1e-6 * cnt_pred_small; o = np.argsort(s)
def rep(tag, a):
    ap = mix_ap(menu, a); L = c[a].mean() + R; sh = np.bincount(a, minlength=3) / I
    print(f"{tag:46s} AP {ap:.4f} avg {L:.3f} shares {np.round(sh,3)} bar {F(L)+0.003:.4f} AP-F {ap-F(L):+.4f} AP-bar {ap-F(L)-0.003:+.4f}", flush=True)
    return ap - F(L)
for q, qL in ((0.33, 0.223), (0.50, 0.399)):
    print(f"== q512={q} qL={qL}")
    a = np.ones(I, int); a[o[:int(q * I)]] = 0; a[o[I - int(qL * I):]] = 2; ma = rep("(a) exchange rule", a)
    b = np.ones(I, int); b[o[:int(q * I)]] = 0; mb = rep("(b) shrink only (low count -> 512)", b)
    d = np.ones(I, int); d[o[I - int(qL * I):]] = 2; md = rep("(d) L only (high count -> L), +router", d)
    mc = []
    for k in range(3):
        cc = b.copy(); rest = np.where(b == 1)[0]; rs = np.random.RandomState(7 + k).choice(rest, int(qL * I), replace=False); cc[rs] = 2
        mc.append(rep(f"(c) shrink by count + random L, draw {k}", cc))
    e = []
    for k in range(3):
        ee = np.ones(I, int); ee[np.random.RandomState(70 + k).choice(I, int(qL * I), replace=False)] = 2
        e.append(rep(f"(e) random L only, draw {k}", ee))
    print(f"   margins over F: a {ma:+.4f}; b {mb:+.4f}; d {md:+.4f}; b+d-(router counted twice) {mb+md+0.0102*R:+.4f}; c mean {np.mean(mc):+.4f}; e mean {np.mean(e):+.4f}")
