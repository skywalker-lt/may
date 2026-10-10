"""T4 bars for the head route (est. costs; unset basis, F(L)=0.5261+0.0102(L-5.36)). Branch B = o2m head + NMS + 0.19 ms
router (o2o class map count) - 0.05 ms (o2m head cheaper than o2o + TopK, Phase 1). Host NMS 1.15 ms = T4 measured for the
NMS-head families (1.1-1.2). In-engine NMS on the T4 is unmeasured: 0.25 / 0.50 ms est."""
F = lambda L: 0.5261 + 0.0102 * (L - 5.36)
for lab, c in (("in-engine NMS est. 0.25", 0.25), ("in-engine NMS est. 0.50", 0.50), ("host NMS 1.15 (T4 family)", 1.15)):
    for s in (0.2, 0.4, 0.6, 0.8):
        L = 5.36 + s * (c + 0.14); print(f"{lab:27s} share {s:.1f}: avg {L:.3f} ms, bar {F(L) + 0.003:.4f} (dense M +{F(L) + 0.003 - 0.5261:.4f})")
    L = 5.31 + c; print(f"{lab:27s} dense o2m everywhere (no router): {L:.3f} ms, bar {F(L) + 0.003:.4f}, F {F(L):.4f}; 0.5322 - F = {0.5322 - F(L):+.4f}")
