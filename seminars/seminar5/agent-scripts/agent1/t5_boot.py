# paired bootstrap over val2017 images of (routed mixture - dense M@640), for the shrink routes
import numpy as np, mixlib as M, feats as F, contextlib, io, sys
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dumpml_yolo26m_768_coco")]
ids, ns, nm, nl, minA = F.gt_counts(); Xn, Xm = F.features(ids); I = len(ids)
pn_s = F.oof_ridge(Xn, np.log1p(ns), 10); pn_sm = F.oof_ridge(Xn, np.log1p(ns + nm), 10)
rng = np.random.RandomState(5)
def route(hi, lo, s5, s7):
    ch = np.ones(I, int); o = np.lexsort((rng.rand(I), -hi)); ch[o[:int(round(s7 * I))]] = 2
    rest = np.where(ch == 1)[0]; o2 = rest[np.lexsort((rng.rand(len(rest)), lo[rest]))]; ch[o2[:int(round(s5 * I))]] = 0; return ch
S = np.stack([s["ev"] for s in srcs]); E = M.base_eval()
def ap_idx(ch, idx):
    mixed = S[ch[idx], :, :, idx]; mixed = np.transpose(mixed, (1, 2, 0)).reshape(-1)
    E.evalImgs = list(mixed); E.params.imgIds = [ids[i] for i in idx]; E._paramsEval = E.params
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0]
B = int(sys.argv[1]) if len(sys.argv) > 1 else 30
for s5, s7 in ((0.4, 0.0), (0.5, 0.0), (0.4, 0.1)):
    ch = route(pn_s, pn_sm, s5, s7); dense = np.ones(I, int); full = np.arange(I)
    d0 = ap_idx(ch, full) - ap_idx(dense, full); L = s5 * 3.47 + (1 - s5 - s7) * 5.05 + s7 * 7.12; need = 0.0102 * (L - 5.05) + 0.003
    br = np.random.RandomState(0); ds = []
    for b in range(B):
        idx = br.randint(0, I, I); ds.append(ap_idx(ch, idx) - ap_idx(dense, idx))
    ds = np.array(ds)
    print(f"s512 {s5} s768 {s7} avg {L:.2f}: routed - dense640 = {d0:+.4f}; bootstrap sd {ds.std(ddof=1):.4f}; bar requires {need:+.4f}; margin {d0-need:+.4f} = {(d0-need)/ds.std(ddof=1):.1f} sd; P(boot delta < need) {np.mean(ds < need):.2f}", flush=True)
