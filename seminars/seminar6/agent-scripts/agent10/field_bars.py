# Round 2: the bar (hull + 0.003) for each surviving construct under three comparators (T4, est.):
#  A: measured YOLO26 hull; B: A plus LW-DETR-M/L at the published T4 ms x1.10 (M 0.525 pub., L 0.561 measured on this volume);
#  C: B plus LW-DETR-L at 512 (pred. 0.541 on full val from the 500-image paired subset, at est. 5.7 ms); D: LW-DETR-L at x1.50 (unfused attention). The lens-5 rect hull (est. +0.009 at 5.3 ms) is listed apart.
import numpy as np
Y = [(1.65,0.4060),(2.75,0.4795),(3.33,0.4937),(3.78,0.5063),(4.28,0.5117),(4.88,0.5262),(5.32,0.5329),(6.89,0.5417),(12.41,0.5691)]
def hull(pts):
    pts = sorted(pts); h = []
    for p in pts:
        while len(h) >= 2 and (h[-1][1]-h[-2][1])*(p[0]-h[-2][0]) <= (p[1]-h[-2][1])*(h[-1][0]-h[-2][0]): h.pop()
        h.append(p)
    return h
def E(H, t):
    xs, ys = zip(*H); return float(np.interp(t, xs, ys))
A = hull(Y); B = hull(Y + [(4.84,0.525),(7.59,0.561),(14.3,0.583)]); C = hull(Y + [(4.84,0.525),(7.59,0.561),(14.3,0.583),(5.7,0.541)]); D = hull(Y + [(4.84,0.525),(10.35,0.561),(14.3,0.583)])
rows = [("lens 1 QR-64 (L544 + refiner)", 5.67, 0.5359), ("lens 9 RTR (L512 + re-scorer)", 5.38, 0.5315), ("lens 2 FBR share 0.3 (proxy, glue)", 5.48, 0.5381),
        ("lens 2 FBR share 0.5 (proxy, glue)", 5.88, 0.5400), ("lens 8 L512 + R16", 5.79, 0.5340), ("lens 5 ISO-296 (avg ms)", 5.01, 0.5401),
        ("lens 5 ISO-296 (worst ms)", 5.32, 0.5401), ("lens 10 QR-2 (L512 + 2 dec. layers)", 5.68, 0.5335), ("dense YOLO26-L at 544", 5.32, 0.5329), ("dense YOLO26-L at 576", 6.25, 0.5368), ("dense YOLO26-L at 640", 6.89, 0.5417), ("lens 9 RTR at X-fidelity ceiling", 5.38, 0.5440), ("lens 8 L544 + R16 ceiling", 6.23, 0.5426)]
print(f"{'construct':38s} {'ms':>5s} {'pred.':>6s} | bar A | bar B | bar C | bar D | pred.-A | pred.-C")
for n, t, a in rows:
    bA, bB, bC, bD = E(A,t)+0.003, E(B,t)+0.003, E(C,t)+0.003, E(D,t)+0.003
    print(f"{n:38s} {t:5.2f} {a:6.4f} | {bA:.4f} | {bB:.4f} | {bC:.4f} | {bD:.4f} | {a-bA:+.4f} | {a-bC:+.4f}")
print("hull B:", B); print("hull C:", C)
