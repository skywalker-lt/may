# Router cost lever: N-stem thumbnail at 192 / 256 / 320 (T4 cost ~ pixels: est. 0.06 / 0.11 / 0.178 ms), OOF ridge on val, shrink at
# share 0.5 / 0.6, exact mixture AP. Torch N layers 0-5 (checked against the npz n320 at 320).
import os, sys, numpy as np, cv2, torch
torch.set_num_threads(1); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1"); sys.path.insert(0, os.path.dirname(__file__))
import mixlib as M, feats as F
from n320feat import letterbox
net = torch.load("/data/yolo-quant-work/weights/yolo26n.pt", map_location="cpu", weights_only=False)["model"].float().eval().fuse()
def stem(x):
    y = []
    for i, m in enumerate(net.model[:6]):
        x = m(x) if m.f == -1 else m([x if j == -1 else y[j] for j in m.f]); y.append(x)
    return x.mean((2, 3))[0].numpy()
ids, ns, nm, nl, _ = F.gt_counts(); Xn, _ = F.features(ids); I = len(ids)
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco")]
feat = {s: np.zeros((I, 128)) for s in (192, 256, 320)}
with torch.no_grad():
    for k, i in enumerate(ids):
        im = cv2.imread(f"/data/datasets/coco/images/val2017/{int(i):012d}.jpg")
        for s in feat:
            x = torch.from_numpy(np.ascontiguousarray(letterbox(im, s)[..., ::-1].transpose(2, 0, 1)[None])).float() / 255; feat[s][k] = stem(x)
        if k == 50: print("check vs npz n320: max abs diff", float(np.abs(feat[320][:50] - Xn[:50]).max()), flush=True)
rng = np.random.RandomState(3)
for s, X in feat.items():
    p = F.oof_ridge(X, np.log1p(ns + nm), 10); out = []
    for sh in (0.5, 0.6):
        ch = np.ones(I, int); o = np.lexsort((rng.rand(I), p)); ch[o[:int(round(sh * I))]] = 0; out.append(M.score(srcs, ch)[0])
    from scipy.stats import spearmanr
    print(f"N stem @{s}: spearman {spearmanr(p, np.log1p(ns + nm))[0]:.3f} | RS-2 share 0.5 {out[0]:.4f} share 0.6 {out[1]:.4f} | router est. {0.178 * (s / 320) ** 2:.3f} ms T4", flush=True)
