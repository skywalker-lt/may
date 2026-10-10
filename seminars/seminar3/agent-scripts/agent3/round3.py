"""Agent 3, round 3: recompute route statistics from the T4 logits and the probability table."""
import numpy as np
rng = np.random.default_rng(0)
z = np.load('../../inputs/receipts_round2_requests/route_logits.npz')
f16, f32, m16, m32 = z['fp16'], z['fp32'], z['mirror16'], z['mirror32']
def H(c): p = np.asarray(c, float); p = p / p.sum(); p = p[p > 0]; return float(-(p * np.log2(p)).sum())
a16, a32 = f16.argmax(1), f32.argmax(1)
s = np.sort(f32, 1); mar = s[:, -1] - s[:, -2]; mar23 = s[:, -2] - s[:, -3]
flip = a16 != a32
print('flips', flip.sum(), 'margins min/med/max', mar[flip].min(), np.median(mar[flip]), mar[flip].max())
err = f16 - f32
print('max |logit err|', np.abs(err).max(), 'mean', np.abs(err).mean(), 'max |pairwise diff err|', max(np.abs((err[:, i] - err[:, j])).max() for i in range(4) for j in range(i)))
for t in (0.00901, 0.01, 0.02, 0.035, 0.05): print('share margin <=', t, (mar <= t).mean(), 'flip rate inside', flip[mar <= t].mean())
h32, h16 = np.bincount(a32, minlength=4), np.bincount(a16, minlength=4)
print('hist fp32', h32, 'bits', H(h32), 'fp16', h16, 'bits', H(h16), 'smallest imgs/epoch', 118287 * h32.min() / 5000)
bias = err.mean(0); a16c = (f16 - bias).argmax(1)
print('flips after per-expert offset', (a16c != a32).sum())
print('mirror same top1 fp32', (a32 == m32.argmax(1)).mean(), 'fp16', (a16 == m16.argmax(1)).mean())
mm = np.sort(m32, 1); 
same = a32 == m32.argmax(1)
print('mirror-unstable share by margin: <0.05', (~same)[mar < 0.05].mean(), '>=0.05', (~same)[mar >= 0.05].mean(), '>=0.1', (~same)[mar >= 0.1].mean())
print('logit shift under mirror: mean |d margin vec|', np.abs((f32 - f32.mean(1, keepdims=True)) - (m32 - m32.mean(1, keepdims=True))).mean(), 'std of logits across images', f32.std(0))
t2 = lambda x: np.sort(np.argsort(x, 1)[:, -2:], 1)
p32, p16, pm = t2(f32), t2(f16), t2(m32)
pid = p32[:, 0] * 4 + p32[:, 1]; cnt = np.unique(pid, return_counts=True)[1]
print('pair hist', cnt, 'bits', H(cnt), 'min pair share', cnt.min() / 5000, 'set flips fp16', (p32 != p16).any(1).sum(), 'mirror same set', (p32 == pm).all(1).mean())
# kernel exposure in A6: share of images in which expert e is one of the pair
print('A6 expert exposure', [(p32 == e).any(1).mean() for e in range(4)])
# bars
F = lambda L: 0.5261 + 0.0102 * (L - 5.36)
print('bars: packet A', F(5.181) + .003, 'A6', F(5.175) + .003, 'twin A', 0.5261 + 0.0102 * (5.181 - 5.206) + .003, 'twin A with 0.5259', 0.5259 + 0.0102 * (5.181 - 5.206) + .003, 'twin A6', 0.5261 + 0.0102 * (5.175 - 5.206) + .003, 'dense-in-If packet', F(5.206) + .003, 'plain', F(5.36) + .003, 'B', F(5.721) + .003, 'C s+s', F(6.044) + .003, 'D top2', F(6.130) + .003)
# probabilities: split normals, low/high = p10/p90, control and delta independent; observed diff adds seed noise 0.0021
def sn(lo, c, hi, n=400000):
    u = rng.standard_normal(n); return c + np.where(u < 0, (c - lo) / 1.2816, (hi - c) / 1.2816) * u
C = {'80': (0.516, 0.522, 0.527), '600': (0.506, 0.5145, 0.523)}
D = {'A-tied': {'80': (-.004, 0, .004), '600': (-.007, 0, .006), 'bars': (0.5273, 0.5288)},
     'A6': {'80': (-.003, .0005, .004), '600': (-.006, .001, .007), 'bars': (0.5272, 0.5288)},
     'A-full': {'80': (-.005, -.0005, .004), '600': (-.012, -.003, .006), 'bars': (0.5273, 0.5288)},
     'B': {'80': (-.002, 0, .002), '600': (-.003, .001, .004), 'bars': (0.5328, 0.5328)},
     'C s+s': {'80': (-.002, .0005, .003), '600': (-.003, .001, .005), 'bars': (0.5361, 0.5361)},
     'D top-2': {'80': (-.004, .001, .005), '600': (-.005, .002, .008), 'bars': (0.5370, 0.5370)},
     'E in-If': {'80': (0, 0, 0), '600': (0, 0, 0), 'bars': (0.5275, 0.5288)}}
for k, d in D.items():
    for r in ('80', '600'):
        c = sn(*C[r]); dl = sn(*d[r]) if any(d[r]) else np.zeros_like(c); ap = c + dl
        obs = dl + 0.0021 * rng.standard_normal(len(c))
        q = np.percentile(ap, [10, 50, 90])
        print(f"{k:8s} {r:>3s} AP p10/50/90 {q[0]:.4f}/{q[1]:.4f}/{q[2]:.4f} P(packet {d['bars'][0]}) {(ap >= d['bars'][0]).mean():.3f} P(twin {d['bars'][1]}) {(ap >= d['bars'][1]).mean():.3f} P(true d>=.003) {(dl >= .003).mean():.3f} P(obs d>=.003) {(obs >= .003).mean():.3f} joint(twin & true) {((ap >= d['bars'][1]) & (dl >= .003)).mean():.3f}")
# attention MatMul hand count and held parameters
P, w39, rtr = 20411132, 6946816, 33092
print('attn MatMul G', 2 * 4 * 400 * 400 * (32 + 64) / 1e9, 'A-tied held shared-bias', P + 3 * w39 + rtr, 'per-branch-bias', P + 3 * (w39 + 12800) + rtr, 'A6 stored', P + 5 * w39 + rtr, 'A-full', 3580416 + 4 * (15373696 + 1457020) + rtr)
for n, b, held in (('A-tied', 86812852, P + 3 * w39 + rtr), ('A-full', 146015756, 3580416 + 4 * (15373696 + 1457020) + rtr), ('A6', 116.6e6, P + 5 * w39 + rtr)):
    print(n, 'engine bytes - 2*held (MB)', (b - 2 * held) / 1e6)
