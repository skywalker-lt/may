# Margins of the routed P3 drop (branch B = P4/P5 only) on dense L at 512 / 576 and on M at 640.
# AP: exact mixtures (mix_lvl_*.log, round-1 routes_noP3.log); router m640 (post-backbone, free; for L it stands
# in for L's own pooled backbone, est.). Latency est. (T4 unset): B saves a fraction f of the branch-point model's
# time, f = 0.179 central (M measured-profile 0.187; L scaled by its L4 P3 share), range 0.16-0.19.
# "mid" = 20% of the way from tau16 to tau32 (the untrained one-to-many floor on the M subset, round 2 1b), est.
import re, sys
sys.path.insert(0, '.'); from envelope2 import env, F, tM, tL
def parse(fn):
    out = {}
    for l in open(fn):
        m = re.match(r'(\S+) (noP3_\d+) rungs \[\(0, ([\d.]+)\).*?m640_small ([\d.]+).*?null ([\d.]+)', l)
        if m: out[(m.group(2), float(m.group(3)))] = (float(m.group(4)), float(m.group(5)))
    return out
for tag, base, T, fn in [('L512', 0.5262, tL(512), 'mix_lvl_l512.log'), ('L576', 0.5368, tL(576), 'mix_lvl_l576.log')]:
    d = parse(fn)
    print(f'{tag}: dense AP {base:.4f} at est. {T:.2f} ms; env there {env(T):.4f}')
    print('  share | f | avg ms | AP tau16 / mid / tau32 | bar margin (tau16/mid/tau32) | null+0.003 (tau16/tau32) | env+0.003 (tau16/mid/tau32)')
    for s in [0.3, 0.4, 0.49, 0.6]:
        a16, n16 = d[('noP3_16', s)]; a32, n32 = d[('noP3_32', s)]; mid = a16 - 0.2 * (a16 - a32)
        for f in [0.16, 0.179, 0.19]:
            t = T * (1 - f * s)
            print(f'  {s:.2f} | {f:.3f} | {t:.2f} | {a16:.4f} / {mid:.4f} / {a32:.4f} | '
                  + ' / '.join(f'{x - F(t) - 0.003:+.4f}' for x in (a16, mid, a32)) + ' | '
                  + f'{a16 - n16 - 0.003:+.4f} / {a32 - n32 - 0.003:+.4f} | '
                  + ' / '.join(f'{x - env(t) - 0.003:+.4f}' for x in (a16, mid, a32)))
# M at 640 (round 1, m640 router, share 0.49, B saves 1.00 ms measured-profile est.)
t = 5.36 - 0.49
for nm, x in [('tau16', 0.5251), ('tau32', 0.5187)]:
    print(f'M640 1-bit {nm} share 0.49 at {t:.2f}: bar {x - F(t) - 0.003:+.4f}, env+0.003 {x - env(t) - 0.003:+.4f}')
