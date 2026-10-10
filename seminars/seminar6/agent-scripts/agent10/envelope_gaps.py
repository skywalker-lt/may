# YOLO26 T4 envelope (measured points) vs the attention families at their T4 latency. Published RT-DETR rows are
# converted FPS->ms and also shown with this loop's +8-11% measured-over-published offset (est.).
import numpy as np
env = [(4.28,0.5117),(4.88,0.5262),(5.32,0.5329),(6.89,0.5417),(12.41,0.5691)]   # L448,L512,L544,L640,X640 (T4 measured)
def E(t):
    xs, ys = zip(*env); return float(np.interp(t, xs, ys))
rows = [("yolov12m_sdpa (measured)", 5.53, 0.5234), ("yolov12l_sdpa (measured)", 8.00, 0.5375), ("yolov13l_sdpa (measured)", 10.64, 0.5305),
        ("esmoe_m_sdpa (measured)", 9.29, 0.5292),
        ("RT-DETRv2-S/R18 (published 217 FPS)", 1000/217, 0.481), ("RT-DETRv2-M/R50m (published 145 FPS)", 1000/145, 0.519),
        ("RT-DETRv2-L/R50 (published 108 FPS)", 1000/108, 0.534), ("RT-DETRv2-X/R101 (published 74 FPS)", 1000/74, 0.543),
        ("RT-DETR-R50 Det3 (paper 7.9 ms)", 7.9, 0.524), ("RT-DETR-R50 Det5 (paper 8.8 ms)", 8.8, 0.530)]
print(f"{'model':40s} {'T4 ms':>7s} {'AP':>6s} {'env(t)':>7s} {'gap':>7s} | +10% est. measured: {'ms':>5s} {'env':>6s} {'gap':>7s}")
for n, t, a in rows:
    pub = "published" in n or "paper" in n
    t2 = t * 1.10 if pub else t
    print(f"{n:40s} {t:7.2f} {a:6.3f} {E(t):7.4f} {a-E(t):+7.4f} | {t2:5.2f} {E(t2):6.4f} {a-E(t2):+7.4f}" if pub else f"{n:40s} {t:7.2f} {a:6.3f} {E(t):7.4f} {a-E(t):+7.4f}")
print("envelope slope L640->X640: %.4f AP/ms" % ((0.5691-0.5417)/(12.41-6.89)))
