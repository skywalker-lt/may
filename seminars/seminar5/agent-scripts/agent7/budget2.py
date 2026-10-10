"""S/M/L whole-model mixture at the 5.36 ms average: does a better router feature close the gap to the bar?
Rows: thumbnail n320 (deployable, 0.18 ms); M stem m640 (needs a 1.98 ms M stem the whole-model engine does not run: a bound);
GT descriptors (counts, small count, median area, 80-class histogram: a measured-signal bound, not headroom)."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from headroom import ridge_oof, Xn, prox, I, ids, order, z, n_all, n_small, med_area
from budget import solve, C, F, ROUTER
from mixlib import mix_ap, GT
if True:
    pass
Xm = z["m640"][order].astype(np.float64)
cats = sorted(GT.getCatIds()); H = np.zeros((I, 80))
for i, iid in enumerate(ids):
    for a in GT.loadAnns(GT.getAnnIds(imgIds=int(iid), iscrowd=False)): H[i, cats.index(a["category_id"])] += 1
Xg = np.column_stack([np.log1p(n_all), np.log1p(n_small), np.log1p(med_area), np.log1p(H)])
if __name__ == "__main__":
    names = ["dump_yolo26s_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco"]; c = np.array([C[n] for n in names])
    for tag, X in (("n320", Xn), ("m640 (bound)", Xm), ("n320+m640 (bound)", np.hstack([Xn, Xm])), ("GT descriptors (bound)", Xg)):
        P = np.stack([ridge_oof(X, prox(n)) for n in names], 1); a = solve(P, c, 5.18); sh = np.bincount(a, minlength=3) / I
        ap = mix_ap(names, a); nl = np.mean([mix_ap(names, np.random.RandomState(d).permutation(a)) for d in range(2)]); L = c[a].mean() + ROUTER
        print(f"S/M/L router {tag:24s}: AP {ap:.4f} avg {L:.2f} shares {np.round(sh,3)} null {nl:.4f} bar {F(L)+0.003:.4f} AP-bar {ap-F(L)-0.003:+.4f} AP-null {ap-nl:+.4f}", flush=True)
    # the full menu: capacity x resolution operating points as whole-model branches (T4 standalone unset ms, batchP2/P4 and T4-baselines)
    C2 = dict(C); C2.update({"dumpml_yolo26s_768_coco": 3.62, "dumpml_yolo26m_512_coco": 3.78, "dumpml_yolo26m_768_coco": 6.97})
    menu = ["dump_yolo26n_coco", "dump_yolo26s_coco", "dumpml_yolo26s_768_coco", "dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco", "dumpml_yolo26m_768_coco", "dump_yolo26x_coco"]
    c = np.array([C2[n] for n in menu])
    for tag, X in (("n320", Xn), ("GT descriptors (bound)", Xg)):
        P = np.stack([ridge_oof(X, prox(n)) for n in menu], 1)
        for t in (5.18, 6.71):
            a = solve(P, c, t); sh = np.bincount(a, minlength=len(menu)) / I; ap = mix_ap(menu, a)
            nl = np.mean([mix_ap(menu, np.random.RandomState(d).permutation(a)) for d in range(2)]); L = c[a].mean() + ROUTER
            print(f"menu(8) router {tag:22s} budget {t}: AP {ap:.4f} avg {L:.2f} shares {np.round(sh,3)} null {nl:.4f} bar {F(L)+0.003:.4f} AP-bar {ap-F(L)-0.003:+.4f} AP-null {ap-nl:+.4f}", flush=True)
    Pg = np.stack([prox(n) for n in menu], 1); a = solve(Pg, c, 5.18); L = c[a].mean() + ROUTER
    print(f"menu(8) proxy oracle budget 5.18: AP {mix_ap(menu, a):.4f} avg {L:.2f} shares {np.round(np.bincount(a, minlength=len(menu))/I,3)}", flush=True)
