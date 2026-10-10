# Round 1: YOLO26 T4 envelope (measured TRT-native points, T4-envelope.md and T4-baselines.md) vs attention/DETR bases.
# LW-DETR: AP of L measured on this volume (0.561, lwdetr-work eval_baseline.log); other APs and all LW-DETR/RT-DETR
# latencies are the papers' published T4 TensorRT fp16 figures, shown raw and x1.10 / x1.20 (this loop runs 8-11% over
# published on YOLO11/26 and 14-19% on the YOLOv12 SDPA exports) -- est., not measured.
import numpy as np
env = [(1.65,0.4060),(2.75,0.4795),(3.33,0.4937),(3.78,0.5063),(4.28,0.5117),(4.88,0.5262),(5.32,0.5329),(6.89,0.5417),(12.41,0.5691)]
def hull(pts):
    pts = sorted(pts); h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= (p[1]-h[-2][1])*(h[-1][0]-h[-2][0]): h.pop()
        h.append(p)
    return h
H = hull(env)
def E(t):
    xs, ys = zip(*H); return float(np.interp(t, xs, ys)) if t <= xs[-1] else float("nan")
rows = [("yolov12m_sdpa (measured)", 5.53, 0.5234, 0), ("yolov12l_sdpa (measured)", 8.00, 0.5375, 0),
        ("RT-DETRv2-S R18 (pub. 217 FPS)", 1000/217, 0.481, 1), ("RT-DETRv2-L R50 (pub. 108 FPS)", 1000/108, 0.534, 1),
        ("LW-DETR-tiny (pub.)", 2.0, 0.426, 1), ("LW-DETR-small (pub.)", 2.6, 0.480, 1), ("LW-DETR-medium (pub.)", 4.4, 0.525, 1),
        ("LW-DETR-large (AP here, ms pub.)", 6.9, 0.561, 1), ("LW-DETR-xlarge (pub.)", 13.0, 0.583, 1)]
print("hull points:", H)
print(f"{'base':34s} {'ms':>6s} {'AP':>6s} | gap at x1.00 / x1.10 / x1.20 of the published ms")
for n, t, a, pub in rows:
    gs = [a - E(t*f) for f in ((1.0, 1.1, 1.2) if pub else (1.0,))]
    print(f"{n:34s} {t:6.2f} {a:6.3f} | " + " / ".join(f"{g:+.4f}" for g in gs))
print("slope of the hull above L640: %.4f AP/ms; X640 is the last measured point (12.41 ms)" % ((0.5691-0.5417)/(12.41-6.89)))
