"""Agent 2, round 3: recomputation of everything the convergence tables rely on. CPU only."""
import math, itertools, numpy as np
np.random.seed(0)
# ---- 1. parameters, engine bytes, bars
base, w39, b39, router = 20_411_132, 6_946_816, 12_800, 33_092
tied = base + 3 * w39 + router
pair6 = base + 5 * w39 + router
print(f"held A-tied {tied/1e6:.3f}M (shared bias; with per-expert bias {(tied+3*b39)/1e6:.3f}M)  A6 stored {pair6/1e6:.3f}M  read {(base+router)/1e6:.3f}M")
for name, held, mb in [("tied", tied, 86.812852), ("pair6", pair6, 116.6), ("full", 70.94e6, 146.015756), ("dense4", base, 45.2)]:
    print(f"  engine {name}: measured {mb:.1f} MB, 2 bytes x held = {2*held/1e6:.1f} MB, overhead {mb-2*held/1e6:.1f} MB")
print(f"MACs conv 34.090 G + attention {2*61.44e6/1e9:.3f} G = {34.090+0.12288:.3f} G; GFLOPs {2*34.090:.2f} conv / {2*34.21288:.2f} with attention")
F = lambda L: 0.5261 + 0.0102 * (L - 5.36)
print(f"packet bar A {F(5.182)+.003:.4f}  A6 {F(5.175)+.003:.4f}  dense-in-If own bar {F(5.204)+.003:.4f} (its AP 0.5259: short by {F(5.204)+.003-.5259:.4f})")
print(f"twin bar A: 0.5259+0.0102*(5.186-5.204)+0.003 = {0.5259+0.0102*(5.186-5.204)+.003:.4f}; with twin AP 0.5261 and r1 pair (5.182,5.213): {0.5261+0.0102*(5.182-5.213)+.003:.4f}; A6 (5.175,5.206): {0.5259+0.0102*(5.175-5.206)+.003:.4f}")
print("latency ratios vs twin: A %.4f  A6 %.4f ; vs baseline: twin %.4f A %.4f A6 %.4f" % (5.186/5.204, 5.175/5.206, 5.204/5.362, 5.186/5.362, 5.175/5.359))
# ---- 2. route statistics from the moderator's logits
d = np.load('/data/tmp/ds-yolo/seminar3/inputs/receipts_round2_requests/route_logits.npz')
f16, f32, m16, m32 = (d[k].astype(np.float64) for k in ('fp16', 'fp32', 'mirror16', 'mirror32'))
def marg(x, a=0, b=1):
    s = np.sort(x, 1)[:, ::-1]; return s[:, a] - s[:, b]
a16, a32 = f16.argmax(1), f32.argmax(1)
flip = a16 != a32; mg = marg(f32)
print(f"top-1 agreement {1-flip.mean():.4f} flips {flip.sum()} margins {mg[flip].min():.5f}-{mg[flip].max():.5f} median {np.median(mg[flip]):.5f}")
bias = (f16 - f32).mean(0); print("per-expert bias", np.round(bias, 5), " max |err|", np.abs(f16 - f32).max().round(4))
print("flips after subtracting the per-expert bias from fp16:", int(((f16 - bias).argmax(1) != a32).sum()))
for t in (0.01, 0.02, 0.035, 0.05): print(f"  share fp32 margin < {t}: {(mg < t).mean():.3f}", end="")
print()
H = lambda c: -sum(p * math.log2(p) for p in c / c.sum() if p > 0)
h1 = np.bincount(a32, minlength=4); print("top-1 hist", h1, f"bits {H(h1):.3f} smallest share {h1.min()/5000:.3f} -> {118287*h1.min()/5000:.0f} images/epoch")
top2 = lambda x: np.sort(np.argsort(x, 1)[:, -2:], 1)
p32 = top2(f32); pid = p32[:, 0] * 4 + p32[:, 1]
ph = np.array([(pid == i * 4 + j).sum() for i, j in itertools.combinations(range(4), 2)])
use = np.array([(p32 == e).any(1).mean() for e in range(4)])
print("pair hist", ph, f"bits {H(ph):.3f} smallest pair share {ph.min()/5000:.3f}; kernel-in-pair share", np.round(use, 3), f"-> smallest kernel sees {118287*use.min():.0f} images/epoch")
mflip = a32 != m32.argmax(1)
print(f"mirror: same top-1 fp32 {1-mflip.mean():.3f} fp16 {(a16 == m16.argmax(1)).mean():.3f}; same top-2 set fp32 {(top2(f32) == top2(m32)).all(1).mean():.3f}")
print(f"  mirror flips with fp32 margin > 0.0177 (max fp16 error): {(mflip & (mg > 0.0177)).sum()} of {mflip.sum()}; > 0.05: {(mflip & (mg > 0.05)).sum()}; median margin of mirror-flipped {np.median(mg[mflip]):.3f}")
print(f"  mirror logit shift: mean |d| {np.abs(f32 - m32).mean():.4f}, p95 {np.percentile(np.abs(f32 - m32), 95):.4f} vs fp16 mean |err| {np.abs(f16 - f32).mean():.4f}")
print(f"  ratio mirror flips / fp16 flips {mflip.sum()/flip.sum():.1f}")
# ---- 3. weight decay on an unsupported delta: surviving fraction exp(-wd * sum lr), linear lr to 0.01 lr0, nbs 64
for lab, lr0, ep in [("80 ep upcycle lr0 0.00038", 0.00038, 80), ("600 ep SGD lr0 0.01", 0.01, 600), ("600 ep lr0 0.001", 0.001, 600)]:
    s = 0.505 * lr0 * ep * 118287 / 64; print(f"decay {lab}: sum lr {s:.1f}, surviving fraction {math.exp(-0.0005*s):.3f}")
# ---- 4. probability model: split normals, low/high = p10/p90, control and delta independent, 2M draws
N = 2_000_000
def split(lo, c, hi):
    z = np.random.standard_normal(N); return c + np.where(z > 0, (hi - c) / 1.2816, (c - lo) / 1.2816) * z
ctrl = {"i": (0.516, 0.522, 0.527), "ii": (0.505, 0.5145, 0.522)}
delta = {("A-tied", "i"): (-.004, 0, .0035), ("A-tied", "ii"): (-.007, 0, .005), ("A6", "i"): (-.003, .0005, .004), ("A6", "ii"): (-.006, .001, .006),
         ("A-full", "i"): (-.006, -.001, .003), ("A-full", "ii"): (-.014, -.003, .004), ("dense", "i"): None, ("dense", "ii"): None}
for (opt, reg), dl in delta.items():
    c = split(*ctrl[reg]); dd = split(*dl) if dl else np.zeros(N); ap = c + dd
    obs = dd + 0.0021 * np.random.standard_normal(N)
    q = np.percentile(ap, [10, 50, 90])
    print(f"{opt:7s} ({reg:2s}) AP p10/50/90 {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}  P(>=.5273) {(ap>=.5273).mean():.3f}  P(>=.5288) {(ap>=.5288).mean():.3f}  "
          f"P(true d>=.003) {(dd>=.003).mean():.3f}  P(observe d>=.003, one seed) {(obs>=.003).mean():.3f}  joint(.5288 & d>=.003) {((ap>=.5288)&(dd>=.003)).mean():.3f}")
