"""Attack on CIWS (agent 6): is either same-cost public member a count specialist? Subset AP by thumbnail-predicted count
tercile/half for YOLO26-M vs YOLOv12-M / YOLO11-M (proxy experts), and the half-confusion of the thumbnail and GT partitions."""
import os, sys, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from headroom import I, cnt_pred_all, n_all
from mixlib import mix_ap_subset
A = "dumpml_yolo26m_coco"
o = np.argsort(cnt_pred_all + 1e-9 * np.arange(I)); rank = np.empty(I, int); rank[o] = np.arange(I)
for B in ("dump_yolov12m_sdpa_coco", "dump_yolo11m_coco", "dump_yolo26l_coco", "dumpml_yolo26m_512_coco"):
    row = []
    for nm, m in (("pred sparse half", rank < I // 2), ("pred crowded half", rank >= I // 2), ("pred sparse third", rank < I // 3), ("pred crowded third", rank >= 2 * I // 3)):
        a = mix_ap_subset([A, B], np.zeros(I, int), m); b = mix_ap_subset([A, B], np.ones(I, int), m); row.append(f"{nm}: M {a:.4f} B {b:.4f} B-M {b-a:+.4f}")
    print(B, " | ".join(row), flush=True)
gmed = np.median(n_all); gt_crowd = n_all > gmed; pr_crowd = rank >= I // 2
print(f"half confusion (thumbnail vs GT count median {gmed:.0f}): GT-crowded predicted sparse {np.mean(~pr_crowd[gt_crowd]):.3f}; GT-sparse predicted crowded {np.mean(pr_crowd[~gt_crowd]):.3f}")
print(f"instances held by predicted-crowded half: {n_all[pr_crowd].sum()/n_all.sum():.3f}; by GT-crowded half {n_all[gt_crowd].sum()/n_all.sum():.3f}")
