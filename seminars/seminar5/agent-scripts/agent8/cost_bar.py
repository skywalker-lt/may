"""Bar arithmetic for NMS-on-demand on the T4 (measured branch costs from receipts_phase1; NMS cost c is a parameter)."""
A_ms, B_raw = 5.36, 5.32   # o2o engine (baseline exporter); end2end=False engine median 5.277-5.329 (batchA)
for c, lab in ((1.12, "host NMS, CLI post ms of yolo11m (measured)"), (0.30, "GPU NMS est."), (0.60, "GPU NMS est. high")):
    for s in (0.1, 0.2, 0.3, 1.0):
        L = (1 - s) * A_ms + s * (B_raw + c); bar = 0.5261 + 0.0102 * (L - 5.36) + 0.003
        print(f"{lab:44s} share {s:.1f}: avg {L:.3f} ms, worst {B_raw + c:.2f} ms, bar {bar:.4f}, gain needed over dense M {bar - 0.5261:+.4f}")
