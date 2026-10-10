"""Agent 4, round 3: recomputation of every number relied on in round3/agent4.md (CPU, 2 threads)."""
import json, math, itertools, numpy as np, torch
torch.set_num_threads(2)
R = "/data/tmp/ds-yolo/seminar3/"
# ---- 1. counts
c = json.load(open(R + "work/agent4/counts.json"))
P = sum(c["pl"].values()); M = sum(c["ml"].values())
att = 2 * 4 * 400 * 400 * 96
print(f"[counts] YOLO26-M fused params {P}; conv MACs {M/1e9:.4f} G; attention MatMul {att/1e9:.4f} G; total {(M+att)/1e9:.4f} G; GFLOPs conv {2*M/1e9:.2f}, with attention {2*(M+att)/1e9:.2f}")
try:
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.modules.moe.weight_bank import WeightBankConv2d  # noqa
except Exception as e:
    WeightBankConv2d = None; print("[counts] import", repr(e)[:120])
try:
    m = DetectionModel("yolo26m-wb.yaml", ch=3, nc=80, verbose=False).eval(); m.fuse(verbose=False)
    tot = sum(p.numel() for p in m.parameters())
    ex = sum(p.numel() for n, p in m.named_parameters() if ".experts." in n)
    rt = sum(p.numel() for n, p in m.named_parameters() if "router" in n)
    nb = sum(1 for n, mod in m.named_modules() if hasattr(mod, "experts"))
    bb = sum(mod.bias.numel() for n, mod in m.named_modules() if hasattr(mod, "experts") and getattr(mod, "bias", None) is not None)
    print(f"[counts] yolo26m-wb fused: total {tot}; banks {nb}; expert weights {ex} (per expert {ex//4}); bank biases (shared) {bb}; router {rt}")
    k1 = ex // 4
except Exception as e:
    print("[counts] wb build failed", repr(e)[:200]); k1 = 6946816
held4 = P + 3 * k1 + 33092
print(f"[counts] A-tied held, shared bias: {held4}; six pair branches stored: {P + 5*k1 + 33092}; fully distinct: see round 2 (70.94M)")
for name, held, meas in [("if_dense4", P, 45.2), ("wb_top1_if", held4, 86.8), ("if_pair6", P + 5 * k1 + 33092, 116.6), ("if_distinct4", 70.94e6, 146.0)]:
    print(f"[engine] {name}: 2 bytes x held = {2*held/1e6:.1f} MB; measured {meas}; overhead {meas-2*held/1e6:.1f} MB")
# ---- 2. bars
F = lambda L: 0.5261 + 0.0102 * (L - 5.36)
print(f"[bar] packet A-tied F(5.182)+0.003 = {F(5.182)+0.003:.4f}; A6 F(5.175)+0.003 = {F(5.175)+0.003:.4f}")
for nm, L, Ld in [("A-tied r2 rows", 5.182, 5.213), ("A-tied r3 rows", 5.186, 5.204), ("A6 r3 rows", 5.175, 5.206)]:
    print(f"[bar] like-for-like {nm}: anchor 0.5259 -> {0.5259+0.0102*(L-Ld)+0.003:.4f}; anchor 0.5261 -> {0.5261+0.0102*(L-Ld)+0.003:.4f}; latency ratio {L/Ld:.4f}")
# ---- 3. routing
z = np.load(R + "inputs/receipts_round2_requests/route_logits.npz")
f16, f32, m16, m32 = z["fp16"], z["fp32"], z["mirror16"], z["mirror32"]
def marg(x, a=0, b=1):
    s = np.sort(x, 1)[:, ::-1]; return s[:, a] - s[:, b]
a16, a32 = f16.argmax(1), f32.argmax(1)
fl = a16 != a32; mg = marg(f32)
h = np.bincount(a32, minlength=4) / 5000
print(f"[route] fp32 hist {np.bincount(a32, minlength=4)}; entropy {-(h*np.log2(h)).sum():.3f} bits; smallest share {h.min():.3f} -> {h.min()*118287:.0f} images/epoch")
print(f"[route] top-1 agreement {1-fl.mean():.4f} ({fl.sum()} flips); flipped margins {mg[fl].min():.5f}..{mg[fl].max():.5f}, median {np.median(mg[fl]):.5f}; max |logit err| {np.abs(f16-f32).max():.4f}")
for t in (0.01, 0.02, 0.035, 0.05):
    print(f"[route] share of images with fp32 top-1 margin < {t}: {(mg<t).mean():.4f}; flips among them {fl[mg<t].sum()}")
off = (f16 - f32).mean(0); print(f"[route] per-expert offset {np.round(off,5)}; flips after subtracting it: {((f16-off).argmax(1)!=a32).sum()}")
def top2(x): return np.sort(np.argsort(x, 1)[:, -2:], 1)
s16, s32 = top2(f16), top2(f32); sf = (s16 != s32).any(1)
print(f"[route] top-2 set agreement {1-sf.mean():.4f} ({sf.sum()} flips)")
mf = m32.argmax(1) != a32; d = np.abs(m32 - f32)
print(f"[mirror] same top-1 fp32 {1-mf.mean():.4f}; fp16 {(m16.argmax(1)==a16).mean():.4f}; same top-2 set fp32 {1-(top2(m32)!=s32).any(1).mean():.4f}")
print(f"[mirror] |logit(mirror)-logit(image)| mean {d.mean():.4f}, max {d.max():.3f} vs fp16 error mean {np.abs(f16-f32).mean():.4f}; ratio {d.mean()/np.abs(f16-f32).mean():.0f}x")
print(f"[mirror] margin of mirror-flipped images: median {np.median(mg[mf]):.4f}, p90 {np.quantile(mg[mf],0.9):.4f}; all images median {np.median(mg):.4f}")
for t in (0.02, 0.05, 0.1, 0.2):
    print(f"[mirror] images with margin >= {t}: share {(mg>=t).mean():.3f}, mirror-flip rate among them {mf[mg>=t].mean():.3f}")
pairs = {}
for a, b in zip(a32, m32.argmax(1)):
    if a != b: pairs[(int(a), int(b))] = pairs.get((int(a), int(b)), 0) + 1
print(f"[mirror] flip transitions image->mirror: {dict(sorted(pairs.items()))}")
hm = np.bincount(m32.argmax(1), minlength=4); print(f"[mirror] mirror hist {hm}")
# ---- 4. probability model (predictions; normal model)
N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
rng = np.random.default_rng(0); n = 400000
def model(tag, c, sc, dl, dc, dh, bars, seed=0.0015):
    sd = (dh - dl) / 2.563  # low/high read as p10/p90
    ctrl = rng.normal(c, sc, n); delta = rng.normal(dc, sd, n)
    ap = ctrl + delta + rng.normal(0, seed, n); obs_ctrl = ctrl + rng.normal(0, seed, n)
    out = [f"P(bar {b:.4f})={np.mean(ap>=b):.3f}" for b in bars]
    out.append(f"P(true delta>=0.003)={np.mean(delta>=0.003):.3f}"); out.append(f"P(observe ctrl+0.003, one seed pair)={np.mean(ap-obs_ctrl>=0.003):.3f}")
    out.append(f"joint(like-for-like bar & true delta>=0.003)={np.mean((ap>=bars[-1])&(delta>=0.003)):.3f}")
    q = np.quantile(ap, [0.1, 0.5, 0.9]); print(f"[pred] {tag}: AP p10/p50/p90 {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f}; " + "; ".join(out))
c80, s80, c600, s600 = 0.522, 0.0040, 0.5145, 0.0060
model("dense plain 80 (bar F(5.36)+.003=0.5291)", c80, s80, 0, 0, 1e-9, [0.5291])
model("dense plain 600", c600, s600, 0, 0, 1e-9, [0.5291])
model("dense-in-If 80 (packet bar F(5.204)+.003)", c80, s80, 0, 0, 1e-9, [F(5.204) + 0.003])
model("A-tied 80", c80, s80, -0.004, 0.0, 0.004, [0.5273, 0.5288])
model("A-tied 600", c600, s600, -0.007, 0.0, 0.006, [0.5273, 0.5288])
model("A6 80", c80, s80, -0.003, 0.0005, 0.0045, [0.5272, 0.5286])
model("A6 600", c600, s600, -0.006, 0.001, 0.0065, [0.5272, 0.5286])
model("A-full 80", c80, s80, -0.005, -0.001, 0.004, [0.5273, 0.5288])
model("A-full 600", c600, s600, -0.012, -0.003, 0.006, [0.5273, 0.5288])
model("D top-2 80 (bar 0.5370)", c80, s80, -0.003, 0.001, 0.006, [0.5370, 0.5370])
model("D top-2 600", c600, s600, -0.005, 0.0015, 0.008, [0.5370, 0.5370])
model("B 80 (bar 0.5328)", c80, s80, -0.002, 0.0, 0.003, [0.5328, 0.5328])
model("C s+s 80 (bar 0.5361)", c80, s80, -0.002, 0.0005, 0.0035, [0.5361, 0.5361])
# gate arithmetic: same-checkpoint twin test, paired sigma 0.0013
for true in (0.0, 0.001, 0.003, 0.005):
    print(f"[gate] true twin gap {true}: P(observed gap >= 0.003 | paired sigma 0.0013) = {1-N((0.003-true)/0.0013):.3f}")
# ---- 5. six-branch (top-2 pair) step-0 statistics, fp32
ph = {}
for a, b in s32: ph[(int(a), int(b))] = ph.get((int(a), int(b)), 0) + 1
pp = np.array(list(ph.values())) / 5000
part = [sum(v for k, v in ph.items() if e in k) / 5000 for e in range(4)]
print(f"[pair] hist {dict(sorted(ph.items()))}; entropy {-(pp*np.log2(pp)).sum():.3f} of {math.log2(6):.3f} bits; largest/smallest pair share {pp.max():.3f}/{pp.min():.3f}")
print(f"[pair] share of images whose pair contains expert e: {np.round(part,4)}; weakest -> {min(part)*118287:.0f} images/epoch (top-1 weakest {h.min()*118287:.0f})")
print(f"[bar] dense-in-If packet bar F(5.204)+0.003 = {F(5.204)+0.003:.4f}; ratios: if_dense4/base {5.204/5.362:.4f}, {5.213/5.358:.4f}; A-tied/base {5.182/5.358:.4f}, {5.186/5.362:.4f}; A6/base {5.175/5.359:.4f}")
