"""Agent 10, round 3. Count-routed menus over the measured {M,L} x scale dumps at several T4 averages (unset basis; L-at-scale and
M@448/576/608 costs are est. by agent 7's pixel rule), against (a) F+0.003, (c) the dense envelope + 0.003, (b) share null + 0.003.
Uses agent 7's mixlib/headroom/menus read-only (imports only). ONE thread."""
import os, sys, itertools, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7/r2")
import menus as M7
from mixlib import mix_ap
R = float(os.environ.get("RCOST", "0.18"))
pts, env, F, assign, I = M7.pts, M7.env, M7.F, M7.assign, M7.I
def run(keys, shares, T):
    names = [pts[k][0] for k in keys]; c = np.array([pts[k][1] for k in keys]); a = assign(shares); L = c[a].mean() + R
    ap = mix_ap(names, a)
    nl = np.mean([mix_ap(names, np.random.RandomState(400 + d).permutation(a)) for d in range(2)])
    return L, ap, ap - F(L) - 0.003, ap - env(L) - 0.003, ap - nl - 0.003
def shares_for(keys, T, q=None):
    c = [pts[k][1] for k in keys]
    if len(keys) == 2:
        x = (c[1] + R - T) / (c[1] - c[0]); return [x, 1 - x] if 0 < x < 1 else None
    qe = (q * (c[1] - c[0]) - R - (c[1] - T)) / (c[2] - c[1])
    return [q, 1 - q - qe, qe] if 0 < qe < 1 - q else None
menus2 = [("M@512","M@640"),("M@448","M@640"),("L@448","L@640"),("L@512","L@640"),("L@448","L@576"),("L@512","L@576"),("M@448","L@576"),("M@512","L@640"),("M@448","L@512"),("M@512","L@576")]
menus3 = [("M@512","M@640","L@640"),("M@448","L@512","L@640"),("L@448","L@512","L@640"),("L@448","L@576","L@640"),("M@448","L@512","L@576")]
for T in [float(t) for t in os.environ.get("TARGETS", "4.6 5.0 5.8 6.3").split()]:
    print(f"=== target {T} ms (router {R} ms on every image); env {env(T):.4f}, F+0.003 {F(T)+0.003:.4f}", flush=True)
    rows = []
    for k in menus2:
        sh = shares_for(k, T)
        if sh: rows.append(("/".join(k), np.round(sh, 2), *run(list(k), sh, T)))
    for k in menus3:
        for q in (0.2, 0.33, 0.5):
            sh = shares_for(k, T, q)
            if sh: rows.append(("/".join(k), np.round(sh, 2), *run(list(k), sh, T)))
    for nm, sh, L, ap, da, dc, db in sorted(rows, key=lambda r: -r[6]):
        print(f"  {nm:22s} {str(sh):18s} avg {L:.2f} AP {ap:.4f} | a: {da:+.4f} | c: {dc:+.4f} | b: {db:+.4f}", flush=True)
