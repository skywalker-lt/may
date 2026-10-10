# Agent 1 round 3: recompute counts, routing statistics (route_logits.npz), bars, and the consensus probability model.
import json, itertools, numpy as np
np.set_printoptions(precision=4, suppress=True)
S = '/data/tmp/ds-yolo/seminar3/'
rows = json.load(open(S + 'work/agent1/rows.json'))
print('rows.json type', type(rows).__name__, (list(rows)[:6] if isinstance(rows, dict) else rows[0]))
P, R, RT = 20411132, 6946816, 33092
print('held: A-tied %.3f M  A6 stored %.3f M  bytes rule A-tied %.1f MB  A6 %.1f MB (+4.24)' % (
    (P + 3*R + RT)/1e6, (P + 5*R + RT)/1e6, 2*(P+3*R+RT)/1e6 + 4.24, 2*(P+5*R+RT)/1e6 + 4.24))
print('A6 overhead implied by 116.6 MB: %.2f MB' % (116.6 - 2*(P+5*R+RT)/1e6))
F = lambda L: 0.5261 + 0.0102*(L - 5.36)
for nm, L, Ld in (('A-tied', 5.181, 5.206), ('A6', 5.175, 5.206)):
    print(nm, 'packet bar %.4f  twin bar (0.5261 anchor) %.4f  (0.5259 anchor) %.4f' % (
        F(L) + .003, 0.5261 + .0102*(L-Ld) + .003, 0.5259 + .0102*(L-Ld) + .003))
print('dense twin on packet front: F(5.206)=%.4f, twin AP 0.5259 -> %+.4f above' % (F(5.206), 0.5259 - F(5.206)))
z = np.load(S + 'inputs/receipts_round2_requests/route_logits.npz')
a16, a32, m16, m32 = z['fp16'], z['fp32'], z['mirror16'], z['mirror32']
H = lambda c: float(-(c[c > 0]/c.sum() * np.log2(c[c > 0]/c.sum())).sum())
t16, t32 = a16.argmax(1), a32.argmax(1)
srt = np.sort(a32, 1); m12 = srt[:, -1] - srt[:, -2]; m23 = srt[:, -2] - srt[:, -3]
print('top1 flips', (t16 != t32).sum(), 'hist16', np.bincount(t16, minlength=4), 'bits16 %.3f bits32 %.3f' % (
    H(np.bincount(t16, minlength=4)), H(np.bincount(t32, minlength=4))))
bias = (a16 - a32).mean(0); print('fp16-fp32 mean', bias, 'std', (a16 - a32).std(0), 'max|err| %.4f' % np.abs(a16-a32).max())
print('flips after removing per-expert offset:', ((a16 - bias).argmax(1) != t32).sum())
perr = np.abs((a16 - a32)[:, :, None] - (a16 - a32)[:, None, :]).max((1, 2))
print('pairwise logit error: max %.4f  p99 %.4f' % (perr.max(), np.quantile(perr, .99)))
for th in (0.01, 0.02, 0.035, 0.05, 0.1): print('  share m12<%.3f: %.3f  m23<%.3f: %.3f' % (th, (m12 < th).mean(), th, (m23 < th).mean()))
set2 = lambda a: np.sort(np.argsort(a, 1)[:, -2:], 1)
p32, p16, pm = set2(a32), set2(a16), set2(m32)
print('top2-set flips fp16', (p32 != p16).any(1).sum(), ' pair bits %.3f of 2.585' % H(np.unique(p32[:, 0]*4 + p32[:, 1], return_counts=True)[1]))
mt = m32.argmax(1); same1 = (mt == t32); same2 = (p32 == pm).all(1)
print('mirror: same top1 %.4f  same pair %.4f  pairs share>=1 expert %.4f' % (same1.mean(), same2.mean(),
      np.mean([len(set(x) & set(y)) > 0 for x, y in zip(p32, pm)])))
dm = (m32 - a32); pm_err = np.abs(dm[:, :, None] - dm[:, None, :]).max((1, 2))
print('mirror pairwise logit change: median %.4f p90 %.4f max %.4f ; fp16 median %.4f' % (np.median(pm_err), np.quantile(pm_err, .9), pm_err.max(), np.median(perr)))
for th in (0.02, 0.05, 0.1, 0.2): print('  mirror top1 disagreement among m12>=%.2f: %.3f (n=%d)' % (th, 1 - same1[m12 >= th].mean(), (m12 >= th).sum()))
print('  mirror-flipped images: median m12 %.4f ; share with m12>0.05: %.3f' % (np.median(m12[~same1]), (m12[~same1] > .05).mean()))
# kernel change on a mirror flip: A-tied swaps 1 of 1; A6 changes 0, 1 or 2 of 2 averaged kernels
nshared = np.array([len(set(x) & set(y)) for x, y in zip(p32, pm)])
print('A6 kernels shared with mirror: 2 %.3f  1 %.3f  0 %.3f ; mean changed fraction A6 %.3f vs A-tied %.3f' % (
    (nshared == 2).mean(), (nshared == 1).mean(), (nshared == 0).mean(), (1 - nshared/2).mean(), 1 - same1.mean()))
# decay on delta only: relative shrink exp(-wd_eff * sum lr), cosine-ish mean lr = lr0/2, batch 256 (plan.md), wd 5e-4 x (256/64)
for nm, ep, lr0 in (('80 ep upcycled', 80, 0.00038), ('600 ep scratch', 600, 0.01)):
    s = ep * 118287/256 * lr0/2; print(nm, 'sum lr %.1f  shrink wd5e-4 %.3f  wd2e-3 %.3f' % (s, np.exp(-5e-4*s), np.exp(-2e-3*s)))
# consensus probability model
rng = np.random.default_rng(3); N = 1000000; q = 1.645; se = 0.0015
def sn(lo, c, hi):
    u = rng.standard_normal(N); return np.where(u < 0, c + u*(c-lo)/q, c + u*(hi-c)/q)
ctl = {'80': (0.516, 0.522, 0.527), '600': (0.505, 0.5145, 0.5225)}
dl = {'A-tied': {'80': (-0.005, 0.0, 0.004), '600': (-0.007, 0.0, 0.006)},
      'A6': {'80': (-0.004, 0.0005, 0.0045), '600': (-0.006, 0.001, 0.0065)},
      'A-full': {'80': (-0.007, -0.0005, 0.004), '600': (-0.014, -0.003, 0.006)},
      'D top-2': {'80': (-0.003, 0.001, 0.0055), '600': (-0.005, 0.0015, 0.008)}}
bars = {'A-tied': (0.5273, 0.5288), 'A6': (0.5272, 0.5288), 'A-full': (0.5273, 0.5288), 'D top-2': (0.5370, 0.5370)}
for reg in ('80', '600'):
    c = sn(*ctl[reg]); e1, e2 = rng.normal(0, se, N), rng.normal(0, se, N)
    print(reg, 'dense: p10/50/90', np.quantile(c, [.1, .5, .9]), 'P(>=.5273) %.3f P(>=.5288) %.3f P(plain bar .5291) %.3f null P(obs diff>=.003) %.3f' % (
        np.mean(c >= .5273), np.mean(c >= .5288), np.mean(c >= .5291), np.mean(e2 - e1 >= .003)))
    for nm in dl:
        d = sn(*dl[nm][reg]); a = c + d + (e2 - e1); b1, b2 = bars[nm]
        print('  %-8s p10/50/90 %s P(packet) %.3f P(twin) %.3f P(obs d>=.003) %.3f P(true d>=.003) %.3f joint(twin & obs) %.3f' % (
            nm, np.quantile(c + d, [.1, .5, .9]).round(4), np.mean(c+d >= b1), np.mean(c+d >= b2), np.mean(d + e2 - e1 >= .003), np.mean(d >= .003),
            np.mean((c+d >= b2) & (d + e2 - e1 >= .003))))
# kernel usage share under pair routing (M3 pair histogram recomputed from fp32 logits)
use = np.array([(p32 == e).any(1).mean() for e in range(4)]); print('pair-routing kernel usage share', use, 'smallest -> images/epoch %.0f ; top-1 smallest %.0f' % (use.min()*118287, np.bincount(t32, minlength=4).min()/5000*118287))
