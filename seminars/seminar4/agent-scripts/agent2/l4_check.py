"""Round 4 cross-check (CPU, no torch needed): L4 front, L4 bars, the 39-layer bank's dense time on the L4
from profile_yolo26m.txt, the L-1 cost implied by the micro multipliers, and the L4 M/L share arithmetic.
Run: /data/envs/rtdetr/bin/python l4_check.py  (log: l4_check.log)"""
import re, collections
P = "/data/tmp/ds-yolo/seminar4/inputs/receipts_l4/profiles/"
def prof(name):
    rows = []
    for ln in open(P + name):
        m = re.match(r"\s+([\d.]+)\s+(.*)", ln)
        if m: rows.append((float(m.group(1)), m.group(2)))
    return rows
m = prof("profile_yolo26m.txt"); tot = sum(t for t, _ in m)
# layers 6-22, conv-fused rows; the bank = 39 1x1 convs there. Lower bound: block-level cv1/cv2/cv3 and attn 1x1s (no bottleneck 3x3s)
def lay(n):
    mm = re.search(r"/model\.(\d+)/", n); return int(mm.group(1)) if mm else -1
in_bank_all = [(t, n) for t, n in m if 6 <= lay(n) <= 22 and "Conv" in n and "Reformat" not in n]
bott3x3 = [(t, n) for t, n in in_bank_all if re.search(r"/m\.\d+(/m\.\d+)?/cv[12]/", n) and "attn" not in n]
onebyone = [(t, n) for t, n in in_bank_all if (t, n) not in bott3x3]
print(f"M profile total {tot:.3f} ms; convs in layers 6-22: {len(in_bank_all)} rows {sum(t for t,_ in in_bank_all):.3f} ms; "
      f"bottleneck 3x3-like {len(bott3x3)} rows {sum(t for t,_ in bott3x3):.3f} ms; 1x1-like {len(onebyone)} rows {sum(t for t,_ in onebyone):.3f} ms")
bank = sum(t for t, _ in onebyone)
for mult in (7.5, 9.5):
    print(f"L-1 est. at {mult}x on the 1x1-like rows: M + {(mult-1)*bank:.2f} ms -> {2.551 + (mult-1)*bank:.2f} ms unset-equivalent")
# per-layer glue from the micro graphs: softmoe - dense per layer
print("micro: P3 extra per layer", round(1.565/8 - 0.164/8, 3), "P4 extra per layer", round(1.087/8 - 0.144/8, 3))
print("L-1 est. glue-only: 39 layers x 0.12-0.175 ms =", round(39*0.118, 2), "-", round(39*0.175, 2), "ms over M")
# L4 front (M-L chord, same construction as the packet's T4 front)
Mms, Lms, Mt, Lt = 2.551, 3.346, 5.36, 6.89
sl4 = (0.5417 - 0.5261) / (Lms - Mms); st4 = (0.5417 - 0.5261) / (Lt - Mt)
F = lambda t: 0.5261 + sl4 * (t - Mms)
print(f"L4 front slope {sl4:.4f} AP/ms (T4 {st4:.4f}); 0.003 AP = {0.003/sl4:.3f} ms on the L4, {0.003/st4:.3f} ms on the T4")
for name, t in [("M unset", 2.551), ("M real", 2.42), ("ML engine M branch", 2.50), ("ML L branch (= splice proxy)", 3.05), ("if_scale4 l", 3.12),
                ("M at 768", 3.642), ("L", 3.346), ("attn +0.405", 2.956), ("EsMoE-M sdpa", 4.773), ("soft bank conv", 4.901), ("X", 6.204)]:
    print(f"  bar at {name} {t:.3f} ms: front {F(t):.4f}, bar {F(t)+0.003:.4f}")
# X check of the front: X 0.5691 at 6.204 vs chord
print("X on the chord?", round(F(6.204), 4), "measured 0.5691 ->", "front is concave above L; the M-L chord overstates the bar above L")
# M/L share on the L4
for basis, avg in [("packet (M unset 2.551)", 2.551), ("twin (M real 2.42)", 2.42), ("four-width session M real 2.42, branches 2.58/3.12", None)]:
    if avg is None:
        s = (2.42 - 2.58) / (3.12 - 2.58)
    else:
        s = (avg - 2.50) / (3.05 - 2.50)
    print(f"L4 M/L L-share at {basis}: {s:.3f}; share null {0.5261 + max(s,0)*0.0156:.4f}")
# chord vs front crossing
for t in (2.551, 2.6, 2.7, 2.75, 2.8, 3.05):
    s = (t - 2.50) / 0.55
    print(f"  avg {t:.3f} ms: L share {s:.2f}, chord {0.5261 + s*0.0156:.4f}, chord+router 0.002 {0.5261 + s*0.0156 + 0.002:.4f}, bar {F(t)+0.003:.4f}")
# T4 for comparison (packet basis 5.36, branches 4.85/6.06; twin 5.05)
Ft4 = lambda t: 0.5261 + st4 * (t - 5.36)
for t in (5.36, 5.05):
    s = (t - 4.85) / (6.06 - 4.85)
    print(f"  T4 avg {t:.2f}: L share {s:.2f}, chord {0.5261 + s*0.0156:.4f}, bar {Ft4(t)+0.003:.4f}")
# error table: estimates vs measured
est = {"M on L4 (A1,A2,A4,A5: 3.0-3.5)": (3.0, 3.5, 2.551), "M on L4 (A3: 4x -> ~1.34)": (1.34, 1.34, 2.551),
       "L-1 over M (A2 r2: +1.2-1.4; r3/A1/A5: +1.2-2.0)": (1.2, 2.0, 2.35), "L-2 over M (A1/A5: 1.3-1.5x M = +0.77-1.28; A2 r2 +1.5-2)": (0.77, 2.0, None),
       "L-3 over M (A1 +0.4-0.8, A5 +0.3-0.8)": (0.3, 0.8, None), "768/640 ratio (A1 kill line 1.3)": (1.3, 1.3, 3.642/2.551)}
for k, (lo, hi, meas) in est.items():
    if meas is None: print(f"{k}: measured none (lost / not buildable)"); continue
    print(f"{k}: est {lo}-{hi}, measured {meas:.3f}, error {100*(lo/meas-1):+.0f}% to {100*(hi/meas-1):+.0f}%")
