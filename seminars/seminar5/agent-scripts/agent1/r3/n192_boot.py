# (1) the cheap n192 router (est. 0.064 ms T4) scored for real on the M448/M640 shrink (OOF ridge, val); (2) paired bootstrap over
# val images of routed AP minus the envelope chord at the same est. T4 average (the chord's two dense dumps resampled with it).
import os, sys, numpy as np, cv2, torch, contextlib, io
torch.set_num_threads(1); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1/r3"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1/r2")
import lib3, mixlib as M, feats as F
from n320feat import letterbox
ids, ns, nm, nl, _ = F.gt_counts(); Xn, _ = F.features(ids); I = len(ids)
fp = "/data/tmp/ds-yolo/seminar5/work/agent1/r3/val_n192.npy"
if not os.path.exists(fp):
    net = torch.load("/data/yolo-quant-work/weights/yolo26n.pt", map_location="cpu", weights_only=False)["model"].float().eval().fuse()
    def stem(x):
        y = []
        for m in net.model[:6]:
            x = m(x) if m.f == -1 else m([x if j == -1 else y[j] for j in m.f]); y.append(x)
        return x.mean((2, 3))[0].numpy()
    X = np.zeros((I, 128))
    with torch.no_grad():
        for k, i in enumerate(ids):
            im = cv2.imread(f"/data/datasets/coco/images/val2017/{int(i):012d}.jpg")
            X[k] = stem(torch.from_numpy(np.ascontiguousarray(letterbox(im, 192)[..., ::-1].transpose(2, 0, 1)[None])).float() / 255)
    np.save(fp, X)
X192 = np.load(fp)
y = np.log1p(ns + nm); p192 = F.oof_ridge(X192, y, 10); p320 = F.oof_ridge(Xn, y, 10)
src = {k: lib3.evaluated(lib3.NAMES[k]) for k in ("M448", "M576", "M608", "M640", "L448", "L512", "M512")}
af = (3.78 / 5.36 - 0.64) / (1 - 0.64); T = {"M448": 5.36 * (af + (1 - af) * 0.49), "M576": 5.36 * (af + (1 - af) * 0.81), "M608": 5.36 * (af + (1 - af) * 0.9025), "M640": 5.36, "M512": 3.78, "L448": 6.89 * (af + (1 - af) * 0.49), "L512": 6.89 * (af + (1 - af) * 0.64)}
rng = np.random.RandomState(11)
def ch_for(p, q):
    ch = np.ones(I, int); o = np.lexsort((rng.rand(I), p)); ch[o[:int(round(q * I))]] = 0; return ch
for lo, hi in (("M448", "M640"), ("M448", "M608"), ("M448", "M576")):
    for q in (0.5, 0.6):
        a192 = M.score([src[lo], src[hi]], ch_for(p192, q))[0]; a320 = M.score([src[lo], src[hi]], ch_for(p320, q))[0]
        print(f"{lo}/{hi} q {q}: OOF n320 {a320:.4f} | OOF n192 {a192:.4f} | diff {a192 - a320:+.4f}", flush=True)
S = {k: v["ev"] for k, v in src.items()}; E = M.base_eval()
def ap_idx(srcs, ch, idx):
    St = np.stack([S[k] for k in srcs]); mixed = St[ch[idx], :, :, idx]; mixed = np.transpose(mixed, (1, 2, 0)).reshape(-1)
    E.evalImgs = list(mixed); E.params.imgIds = [ids[i] for i in idx]; E._paramsEval = E.params
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0]
B = int(sys.argv[1]) if len(sys.argv) > 1 else 20
DP = {"M448": (T["M448"], src["M448"]), "L448": (T["L448"], src["L448"]), "L512": (T["L512"], src["L512"])}
for lo, hi, q, rc, p, tag in (("M448", "M608", 0.5, 0.064, p192, "n192"), ("M448", "M640", 0.6, 0.064, p192, "n192"), ("M448", "M608", 0.5, 0.18, p320, "n320")):
    ch = ch_for(p, q); t = rc + q * T[lo] + (1 - q) * T[hi]
    a, b = ("M448", "L448") if t < T["L448"] else ("L448", "L512"); w = (t - T[a]) / (T[b] - T[a])
    zero = np.zeros(I, int); full = np.arange(I)
    def delta(idx): return ap_idx([lo, hi], ch, idx) - ((1 - w) * ap_idx([a], zero, idx) + w * ap_idx([b], zero, idx))
    d0 = delta(full); br = np.random.RandomState(0); ds = np.array([delta(br.randint(0, I, I)) for _ in range(B)])
    print(f"{lo}/{hi} q {q} {tag} rc {rc}: avg est. {t:.2f} ms, chord {a}-{b} w {w:.2f}: routed - envelope = {d0:+.4f}, bootstrap sd {ds.std(ddof=1):.4f} (B={B}); "
          f"(c) margin {d0 - 0.003:+.4f} = {(d0 - 0.003) / ds.std(ddof=1):+.1f} sd; P(boot delta < 0.003) {np.mean(ds < 0.003):.2f}", flush=True)
