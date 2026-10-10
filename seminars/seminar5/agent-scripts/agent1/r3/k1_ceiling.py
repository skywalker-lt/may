# Ceiling of the per-image route over the measured envelope: same menus routed by GROUND-TRUTH counts (S+M for the shrink, all for the hi rung)
import sys, numpy as np; sys.argv = sys.argv[:1]
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1/r3")
exec(open("/data/tmp/ds-yolo/seminar5/work/agent1/r3/k1_envelope.py").read().split('print("\\n# two-rung')[0])
for menu, q, qh in ((("M512", "M640"), 0.5, 0), (("M448", "M640"), 0.6, 0), (("M448", "M608"), 0.5, 0), (("L512", "L640"), 0.5, 0),
                    (("L448", "L576"), 0.6, 0), (("M448", "M640", "L640"), 0.6, 0.1)):
    for tag, lo, hi in (("learned", shrink, count), ("GT rule", gt_sm + 1e-3 * count, gt_c + 1e-3 * count)):
        a, t, wc, e, arg = row(menu, q, qh, 0.064, lo_s=lo, hi_s=hi)
        print(f"{'/'.join(menu):16s} q {q:.2f} qh {qh:.2f} {tag:8s} AP {a:.4f} avg {t:.2f} | (a) {a - Fr(t) - 0.003:+.4f} | (c) env {e:.4f} -> {a - e - 0.003:+.4f} (over env {a - e:+.4f})", flush=True)
