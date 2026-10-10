#!/usr/bin/env python3
"""Round 4 (agent 4): cross-check of the L4 table against the rounds-2/3 first-principles estimates,
the L4 front, the neck share of YOLO26-M's L4 profile, and the price of the L4 designs at the measured micro-graph ratios.
Sources: inputs/receipts_l4/l4.log, l4_p2.log, l4_micro.log, profiles/profile_yolo26m.txt; round4/moderator_insert.md;
inputs/T4-baselines.md (APs). Run: OMP_NUM_THREADS=2 /data/envs/rtdetr/bin/python l4_check.py > l4_check.log
"""
import re, numpy as np
R = "/data/tmp/ds-yolo/seminar4/inputs/receipts_l4"
# 1. L4 front from the YOLO26 family (L4 ms from the insert, AP multi-label from T4-baselines.md)
fam = {"n": (0.917, 1.65, 0.4060), "s": (1.344, 2.75, 0.4795), "m": (2.551, 5.37, 0.5261), "l": (3.346, 6.89, 0.5417), "x": (6.204, 12.41, 0.5691)}
print("L4/T4 ratio per scale:", {k: round(v[0] / v[1], 3) for k, v in fam.items()})
sl_ml = (fam["l"][2] - fam["m"][2]) / (fam["l"][0] - fam["m"][0]); sl_lx = (fam["x"][2] - fam["l"][2]) / (fam["x"][0] - fam["l"][0])
sl_t4 = (fam["l"][2] - fam["m"][2]) / (fam["l"][1] - fam["m"][1])
print(f"L4 front slope M->L {sl_ml:.4f} AP/ms, L->X {sl_lx:.4f} AP/ms; T4 slope M->L {sl_t4:.4f} (packet 0.0102)")
def F_l4(t):  # piecewise chord through the measured points, M..L..X
    if t <= fam["l"][0]: return fam["m"][2] + sl_ml * (t - fam["m"][0])
    return fam["l"][2] + sl_lx * (t - fam["l"][0])
for t in (2.42, 2.551, 2.9, 3.05, 3.346, 3.642, 3.9, 4.43, 4.773, 6.204):
    print(f"  L4 front at {t:.3f} ms: {F_l4(t):.4f}, bar +0.003 = {F_l4(t)+0.003:.4f}")
# 2. neck share of M's L4 profile
rows = [(float(a), b) for a, b in re.findall(r"^\s+([0-9.]+)\s+(.*)$", open(f"{R}/profiles/profile_yolo26m.txt").read(), re.M)]
tot = sum(a for a, _ in rows)
def idx(name):
    m = re.search(r"/model\.(\d+)/", name); return int(m.group(1)) if m else -1
neck = [(a, b) for a, b in rows if 11 <= idx(b) <= 22]; head = [(a, b) for a, b in rows if idx(b) == 23]
stem = [(a, b) for a, b in rows if 0 <= idx(b) <= 10]; other = [(a, b) for a, b in rows if idx(b) < 0]
conv_neck = [(a, b) for a, b in neck if "Conv" in b]
print(f"M L4 profile total {tot:.3f} ms over {len(rows)} layers: stem(0-10) {sum(a for a,_ in stem):.3f}, neck(11-22) {sum(a for a,_ in neck):.3f} "
      f"({len(neck)} layers; conv layers {len(conv_neck)}, {sum(a for a,_ in conv_neck):.3f} ms), head(23) {sum(a for a,_ in head):.3f}, other {sum(a for a,_ in other):.3f}")
# 3. price L4-1 (K=4 per-pixel soft mixture) at the measured per-layer ratio: extra = (r-1) x dense layer time, r = 7.5 (P4) .. 9.5 (P3)
neck_conv_ms = sum(a for a, _ in conv_neck); M = 2.551
for r in (7.5, 8.5, 9.5):
    print(f"L4-1 all neck convs at K=4, ratio {r}: est. {M + (r-1)*neck_conv_ms:.2f} ms ({(M + (r-1)*neck_conv_ms)/M:.2f}x M)")
# 1x1 convs only in the neck (cv1/cv2/cv3 and m.* cv1 are 1x1 in A2C2f/C3k2; the 3x3s are m.*cv2 and model.17/20 downsample)
one = [(a, b) for a, b in conv_neck if not re.search(r"/m\.\d+/.*cv2/|model\.(17|20)/conv", b)]
one_ms = sum(a for a, _ in one)
print(f"neck 1x1-class convs: {len(one)} layers, {one_ms:.3f} ms; L4-1 on those only at 7.5x/9.5x: {M+6.5*one_ms:.2f} / {M+8.5*one_ms:.2f} ms")
# minimal form: one K=4 block per level (3 layers, P3/P4/P5 ~ 0.196/0.136/0.10 extra each)
print(f"L4-1 minimal (one mixed 1x1 per level): est. {M + (0.196-0.021) + (0.136-0.018) + 0.08:.2f} ms")
# micro-graph decomposition of the P3 soft-mixture layer from the profile
mp = [(float(a), b) for a, b in re.findall(r"^\s+([0-9.]+)\s+(.*)$", open(f"{R}/profiles/profile_l4micro_softmoe4_p3.txt").read(), re.M)]
cat = {}
for a, b in mp:
    k = "expert conv (4x MACs)" if re.match(r"/Conv_(1|3|5|7|9|11|13|15)$", b) else "gate conv" if re.match(r"/Conv(_\d+)?$", b) else \
        "ReduceSum" if "ReduceSum" in b else "Mul" if b.startswith("PWN(/Mul") else "Reformat" if "Reformatting" in b else "Softmax" if "Softmax" in b else "SiLU/other"
    cat[k] = cat.get(k, 0) + a
print("softmoe4_p3 per-layer decomposition (ms per layer, 8 layers):", {k: round(v / 8, 4) for k, v in cat.items()}, "dense P3 layer 0.021")
# 4. crop-pass estimate from the 512/640/768 rows (time = a + b * pixel units, 640^2 = 1)
pts = {0.64: 2.046, 1.0: 2.551, 1.44: 3.642}
for (p1, p2) in ((0.64, 1.0), (0.64, 1.44), (1.0, 1.44)):
    b = (pts[p2] - pts[p1]) / (p2 - p1); a = pts[p1] - b * p1
    print(f"fit on {p1}/{p2}: floor {a:.2f} ms, {b:.2f} ms per 640^2 pixels -> batch-4 160px crop pass (0.25 units) est. {a + 0.25*b:.2f} ms; 640->768 difference 1.09 ms")
# 5. share arithmetic
print("ML on L4: M branch 2.50, L branch 3.05, dense M real 2.42 / unset 2.551 -> L share at real", (2.42-2.50)/0.55, "at unset", round((2.551-2.50)/0.55, 3))
print("ML on T4 (MI3 four-width session): share", round((5.33-4.95)/1.22, 3), "; MI2 two-branch vs real M 5.05:", round((5.05-4.85)/1.21, 3))
print("C on T4 (last receipts, real inputs): 512 br 3.47, 768 br 7.115, M 5.05 -> share", round((5.05-3.47)/(7.115-3.47), 3), "; at 5.36:", round((5.36-3.47)/(7.115-3.47), 3))
print("C on L4 (est., T4 branch ratios x measured L4/T4 0.50): 512 br", round(3.47*0.5, 2), "768 br", round(7.115*0.5, 2), "M 2.42 -> share", round((2.42-1.735)/(3.558-1.735), 3))
# 6. estimate errors
est = [("M on L4", "A1/A2/A4/A5 3.0-3.5; A5 3.3", 3.25, 2.551), ("N on L4", "A4 1.2-1.4 (floor-bound)", 1.3, 0.917), ("L on L4", "A5 4.2", 4.2, 3.346), ("X on L4", "A5 7.0", 7.0, 6.204),
       ("L4/T4 throughput", "1.9-2x (A1,A2,A4,A5); 4x (A3)", 1.95, 1/0.485), ("EsMoE-M on L4 vs M", "A5: 'its L4 row says how much is glue' (no number)", None, 4.773/2.551)]
for n, who, e, m in est:
    print(f"{n}: stated {who}; measured {m:.3f}" + (f"; error {100*(e-m)/m:+.0f}%" if e else ""))
