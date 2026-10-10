"""Head cost of the exported (one-to-one) YOLO26-M head as a function of vocabulary size.

Shapes read from the public yolo26m.pt (see head_shapes.log): at P3/P4/P5 the cls branch is
DWConv(x,x,3) -> Conv(x,c3,1) -> DWConv(c3,c3,3) -> Conv(c3,c3,1) -> Conv2d(c3, nc, 1), with
c3 = max(ch[0], min(nc, 100)) = 256 for M whatever nc >= 100; the box branch is nc-independent.
Only the final 1x1 scales with nc: MACs = c3 * nc * A, A = 6400 + 1600 + 400 = 8400 anchors at 640.
T4 conversion: YOLO26-M = 34.213 GMAC (seminar-3 table) at 5.36 ms unset => 6.38 GMAC/ms (est.; the
T4 is MAC-bound at M, T4-baselines finding 6). Score traffic: the end2end post-processing reads the
[8400, nc] fp16 score map at least once (max over classes + TopK); charged at 250 GB/s (est.).
"""
A = 8400
c3 = 256
GMAC_PER_MS = 34.213 / 5.36
BW = 250e9  # bytes/s est.

def head_cls_final(nc):
    return c3 * nc * A

rows = []
base = head_cls_final(80)
for name, nc in [("COCO", 80), ("union of the four domains", 82), ("Objects365", 365),
                 ("Open Images", 601), ("LVIS", 1203), ("COCO+O365+LVIS union", 1648),
                 ("ImageNet-21k-style vocabulary", 21000)]:
    macs = head_cls_final(nc)
    d_gmac = (macs - base) / 1e9
    ms_mac = d_gmac / GMAC_PER_MS
    ms_bw = (A * nc * 2) / BW * 1e3  # one read of the fp16 score map, ms
    rows.append((name, nc, macs / 1e9, d_gmac, ms_mac, ms_bw, ms_mac + ms_bw))

print(f"{'vocabulary':34s} {'nc':>6s} {'final 1x1 GMAC':>15s} {'+GMAC vs 80':>12s} {'+ms MAC est.':>13s} {'+ms scores est.':>16s} {'+ms total est.':>15s}")
for r in rows:
    print(f"{r[0]:34s} {r[1]:6d} {r[2]:15.3f} {r[3]:12.3f} {r[4]:13.3f} {r[5]:16.3f} {r[6]:15.3f}")

print()
print("Routed-head arithmetic on the T4 (est.):")
for nc, K, k in [(1203, 8, 2), (1648, 8, 2), (1648, 4, 1), (21000, 16, 2)]:
    dense = [r for r in rows if r[1] == nc][0][6]
    routed_static = dense * k / K + 3 * 0.02   # static gather: k/K of the head + runtime-tensor penalty on 3 levels
    routed_if = dense * k / K + 0.25           # conditional form: branch penalty measured +0.25-0.5 ms
    print(f"  nc={nc:6d} K={K:2d} k={k}: dense head +{dense:.2f} ms; static top-k head +{routed_static:.2f} ms "
          f"(saves {dense-routed_static:.2f} ms = {0.0102*(dense-routed_static):.4f} AP at the front's slope); "
          f"If-form +{routed_if:.2f} ms (saves {dense-routed_if:.2f} ms)")
