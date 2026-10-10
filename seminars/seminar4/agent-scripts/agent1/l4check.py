#!/usr/bin/env python3
"""Round 4 (agent 1): arithmetic on the L4 table (inputs/receipts_l4/l4.log, l4_p2.log, l4_micro.log) and the last T4 receipts.
Command: /data/envs/rtdetr/bin/python l4check.py > l4check.log"""
import numpy as np
T4 = dict(n=1.65, s=2.75, m=5.37, l=6.89, x=12.41); L4 = dict(n=0.917, s=1.344, m=2.551, l=3.346, x=6.204)
AP = dict(n=0.4060, s=0.4795, m=0.5261, l=0.5417, x=0.5691)
print("== L4/T4 ratios:", {k: round(L4[k]/T4[k], 3) for k in T4}, "speedup at M", round(T4["m"]/L4["m"], 2))
sl_t4 = (AP["l"]-AP["m"])/(T4["l"]-T4["m"]); sl_l4 = (AP["l"]-AP["m"])/(L4["l"]-L4["m"])
print(f"== front slope T4 {sl_t4:.4f} AP/ms ({sl_t4*T4['m']:.4f} per 1x M); L4 {sl_l4:.4f} AP/ms ({sl_l4*L4['m']:.4f} per 1x M)")
F4 = lambda t: AP["m"] + sl_l4*(t-L4["m"]); FT = lambda t: AP["m"] + sl_t4*(t-T4["m"])
for k in "nsx": print(f"   L4 front at {k}: {F4(L4[k]):.4f} (actual {AP[k]})   T4 front at {k}: {FT(T4[k]):.4f}")
print("== bars on the L4 (front + 0.003) at candidate latencies")
rows = [("M", 2.551), ("M+0.4 (2 attn layers at P4)", 2.951), ("M+1.2 (my P2 low)", 3.751), ("M+1.5 (12-layer soft mix, micro est.)", 4.051),
        ("L", 3.346), ("M at 768", 3.642), ("P1 at 1.5x M", 3.83), ("EsMoE-M sdpa", 4.773), ("M+4.9 (39-layer soft mix, micro est.)", 7.45), ("X", 6.204)]
for n, t in rows: print(f"   {n:42s} {t:.3f} ms  bar {F4(t)+0.003:.4f}")
print("== EsMoE-M sdpa vs L4 front:", round(0.5292 - F4(4.773), 4), "; vs T4 front:", round(0.5292 - FT(9.29), 4))
print("== micro graphs per layer (ms): dense P3 0.0205 / P4 0.018; soft K=4 P3 0.196 / P4 0.136; attn P4 0.203 (2 layers 0.405)")
ex = dict(p3=0.196-0.0205, p4=0.136-0.018, p5=(0.136-0.018)*0.6)  # P5 est.: 20x20 at C=1024 is MAC-equal to P4 but glue tensors 4x smaller
print("   extra per soft layer: P3 %.3f P4 %.3f P5 est %.3f" % (ex["p3"], ex["p4"], ex["p5"]))
for name, (a, b, c) in {"12 neck 1x1s (4/4/4)": (4, 4, 4), "39 weight-bank 1x1s (13/13/13)": (13, 13, 13), "6 layers at P5 only": (0, 0, 6)}.items():
    add = a*ex["p3"] + b*ex["p4"] + c*ex["p5"]; print(f"   {name:34s} +{add:.2f} ms -> {L4['m']+add:.2f} ms ({(L4['m']+add)/L4['m']:.2f}x M), bar {F4(L4['m']+add)+0.003:.4f}")
# MAC rates in the micro graphs
gm = 256*256*6400/1e9; print(f"== dense P3 1x1: {gm:.2f} GMAC in 0.018-0.0205 ms = {gm/0.0195*1e3/1e3:.1f} TMAC/s = {2*gm/0.0195:.0f} TFLOPS; K=4 conv 1.68 GMAC in 0.052 ms = {2*4*gm/0.052:.0f} TFLOPS (L4 peak 121 dense)")
# resolution rows and the crop-pass bound
px = np.array([0.64, 1.0, 1.44]); t = np.array([2.046, 2.551, 3.642]); b, a = np.polyfit(px, t, 1)
print(f"== L4 time vs pixel units (512/640/768): slope {b:.2f} ms per 640^2, intercept {a:.2f}; batch-4 160px = 0.25 units -> {a+0.25*b:.2f} ms (ideal), 640->768 difference {3.642-2.551:.2f}")
print(f"   768/640 ratio: L4 {3.642/2.551:.3f} T4 {6.96/5.37:.3f}; 512/640: L4 {2.046/2.551:.3f} T4 {3.78/5.37:.3f}")
# C shares
for dev, m, b512, b768 in [("T4 real", 5.05, 3.47, 7.115), ("T4 unset", 5.36, 3.47, 7.115), ("T4 unset (MI3 est. 512=4.45)", 5.36, 4.45, 7.48),
                           ("L4 unset, standalone", 2.551, 2.046, 3.642), ("L4 unset, T4 plumbing ratios", 2.551, 2.046*3.47/3.55, 3.642*7.115/6.49), ("L4 real, T4 plumbing ratios", 2.42, 2.046*3.47/3.55, 3.642*7.115/6.49)]:
    print(f"== C 768 share at M's latency, {dev:32s}: {(m-b512)/(b768-b512):.3f}  (512 {b512:.2f}, 768 {b768:.2f}, M {m})  worst {b768/m:.2f}x")
# ML shares
for dev, m, bm, bl in [("T4 (MI2, M real 5.05)", 5.05, 4.85, 6.06), ("T4 (MI2 branches, M unset 5.36)", 5.36, 4.85, 6.06), ("T4 four-width (M 5.33)", 5.33, 4.95, 6.17),
                       ("L4 if_depth_ml, M real 2.42", 2.42, 2.50, 3.05), ("L4 if_depth_ml, M unset 2.551", 2.551, 2.50, 3.05), ("L4 if_scale4 m/l, M unset", 2.551, 2.58, 3.12)]:
    print(f"== ML L share at M's latency, {dev:34s}: {(m-bm)/(bl-bm):+.3f}  (M branch/dense M = {bm/m:.3f})")
print("== splice H est. on L4: 0.935 x L =", round(0.935*3.346, 3), "bar", round(F4(0.935*3.346)+0.003, 4))
