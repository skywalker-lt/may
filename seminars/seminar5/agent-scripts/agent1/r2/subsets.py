# Per-ms value of depth (L tail) vs pixels (768) vs shrink (512) on thumbnail-predicted scene types (attack on direction 2 / merge with 4, 7)
import sys, io, contextlib; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1")
import numpy as np, mixlib as M, feats as F
names = ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dumpml_yolo26m_768_coco", "dump_yolo26l_coco")
srcs = [M.evaluated(n) for n in names]
ids, ns, nm, nl, _ = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
p_s = F.oof_ridge(Xn, np.log1p(ns), 10); p_l = F.oof_ridge(Xn, np.log1p(nl), 10)
def sub_ap(src, mask):
    E = M.base_eval(); ev = src["ev"][:, :, mask]; E.evalImgs = list(ev.reshape(-1)); E.params.imgIds = list(np.array(src["imgIds"])[mask]); E._paramsEval = E.params
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0]
T = {"512": 3.47, "640": 5.05, "768": 7.12, "L": 6.55}
o_s = np.argsort(-p_s); o_l = np.argsort(-p_l)
for lab, order in (("pred. small-heavy", o_s), ("pred. large-heavy", o_l)):
    for frac in (0.2, 0.33, 0.5):
        mask = np.zeros(I, bool); mask[order[:int(frac * I)]] = True
        for mk, mm in ((f"top {frac:.2f}", mask), (f"rest {1-frac:.2f}", ~mask)):
            a = [sub_ap(s, mm) for s in srcs]
            print(f"{lab:18s} {mk:10s}: M512 {a[0]:.4f} M640 {a[1]:.4f} M768 {a[2]:.4f} L640 {a[3]:.4f} | per real ms over M640: 768 {(a[2]-a[1])/(7.12-5.05):+.4f} (standalone {(a[2]-a[1])/(6.49-5.05):+.4f}), L {(a[3]-a[1])/1.50:+.4f}, shrink saves {(a[1]-a[0])/1.58:.4f} per ms", flush=True)
