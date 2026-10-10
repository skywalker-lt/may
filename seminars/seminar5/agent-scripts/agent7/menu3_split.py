"""Split-half check of the menu3 count rule: q chosen on one half of val2017 (by image-id parity), scored on the other half,
with dense M and the share-matched null on the same half; plus an L4-cost version (est. costs from the L4 table)."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(__file__))
from headroom import I, ids, cnt_pred_small, cnt_pred_all
from mixlib import mix_ap_subset, mix_ap
menu = ["dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco"]
s = cnt_pred_all + 1e-6 * cnt_pred_small
def assign(q, c512, cm, cl, r, mask):
    o = np.where(mask)[0][np.argsort(s[mask])]; n = len(o); qL = max(0.0, (q * (cm - c512) - r) / (cl - cm))
    a = np.ones(I, int); a[o[:int(q * n)]] = 0; a[o[n - int(qL * n):]] = 2; return a, qL
for name, mask in (("even ids", ids % 2 == 0), ("odd ids", ids % 2 == 1)):
    other = ~mask; best = None
    for q in (0.25, 0.33, 0.40, 0.50, 0.55):
        a, _ = assign(q, 3.78, 5.36, 6.89, 0.18, mask); ap = mix_ap_subset(menu, a, mask)
        if best is None or ap > best[1]: best = (q, ap)
    a, qL = assign(best[0], 3.78, 5.36, 6.89, 0.18, other); ap = mix_ap_subset(menu, a, other)
    dense = mix_ap_subset(menu, np.ones(I, int), other)
    nl = np.mean([mix_ap_subset(menu, np.where(other, np.random.RandomState(9 + d).permutation(a), a), other) for d in range(2)])
    print(f"q chosen on {name}: q={best[0]} (tuning-half AP {best[1]:.4f}); held-out half: rule {ap:.4f}, dense M {dense:.4f}, null {nl:.4f}, rule-dense {ap-dense:+.4f}, rule-null {ap-nl:+.4f}", flush=True)
# L4 standalone costs (L4 table): M512 2.046, M 2.551, L 3.346; router est. 0.10 ms (N stem at 320; not measured on the L4)
for q in (0.33, 0.50, 0.55):
    a, qL = assign(q, 2.046, 2.551, 3.346, 0.10, np.ones(I, bool)); ap = mix_ap(menu, a); L = np.array([2.046, 2.551, 3.346])[a].mean() + 0.10
    print(f"L4 est. q512={q} qL={qL:.3f}: AP {ap:.4f} avg {L:.3f} ms; L4 bar F_L4(avg)+0.003 = {0.5261 + 0.0196*(L-2.551) + 0.003:.4f}", flush=True)
for q in (0.55,):
    a, qL = assign(q, 3.78, 5.36, 6.89, 0.18, np.ones(I, bool)); ap = mix_ap(menu, a); sh = np.bincount(a, minlength=3) / I
    nl = np.mean([mix_ap(menu, np.random.RandomState(70 + d).permutation(a)) for d in range(3)]); L = np.array([3.78, 5.36, 6.89])[a].mean() + 0.18
    print(f"T4 unset q512={q}: shares {np.round(sh,3)} AP {ap:.4f} avg {L:.3f} null {nl:.4f} AP-bar {ap-0.5261-0.0102*(L-5.36)-0.003:+.4f} AP-null {ap-nl:+.4f}", flush=True)
