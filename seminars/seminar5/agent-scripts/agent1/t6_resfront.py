# dense resolution front of public YOLO26-M predicted from its three measured scales (quadratic in ln scale; T4 latency = a + b*pixels fitted to the
# two in-engine branch timings 3.47 @512 and dense real 5.05 @640). Prediction only; the 576/608 dumps are the measurement request.
import numpy as np
x = np.log(np.array([512, 640, 768]) / 640); y = np.array([0.5063, 0.5261, 0.5196]); c = np.polyfit(x, y, 2)
b = (5.05 - 3.47) / (1 - 0.64); a = 5.05 - b
lat = lambda s: a + b * (s / 640) ** 2
print(f"latency model: a={a:.3f} ms, b={b:.3f} ms per (640^2 px); 768 predicted {lat(768):.2f} (measured 7.12 in-engine, 6.49 standalone real)")
for s in (448, 480, 512, 544, 576, 608, 640, 672, 704, 768):
    L = lat(s); ap = np.polyval(c, np.log(s / 640)); F = 0.5261 + 0.0102 * (L - 5.05)
    print(f"dense M@{s}: T4 est. {L:.2f} ms, AP pred. {ap:.4f}, front {F:.4f}, AP-front {ap-F:+.4f}")
# routed shrink points (measured on public weights, t4_shrink.log) against the predicted dense resolution front at equal latency
for s5, L, apr in ((0.3, 4.58, 0.5250), (0.4, 4.42, 0.5242), (0.5, 4.26, 0.5230), (0.6, 4.10, 0.5216), (0.7, 3.94, 0.5192)):
    s = 640 * np.sqrt((L - a) / b); apd = np.polyval(c, np.log(s / 640))
    print(f"routed s512={s5}: {L:.2f} ms AP {apr:.4f} | dense-resolution front at equal latency: M@{s:.0f} pred. {apd:.4f} | routed - dense {apr-apd:+.4f}")
