# RS-2L (L at 512/576/640, n320 non-large router, out-of-fold) against F and the continuous envelope. T4 unset, est.
import re, sys
sys.path.insert(0, '.'); from envelope2 import env, F, tL
T = {0: tL(512), 1: tL(576), 2: 6.89}
for l in open('mix_rs2l.log'):
    m = re.match(r'(.+?) rungs \[(.*)\]: GT_nonlarge ([\d.]+).*?n320_nonlarge ([\d.]+).*?null ([\d.]+)', l)
    if not m: continue
    rungs = [(int(a), float(b)) for a, b in re.findall(r'\((\d), ([\d.]+)\)', m.group(2))]
    t0 = sum(T[i] * s for i, s in rungs); ap, nl = float(m.group(4)), float(m.group(5))
    cells = []
    for r in (0.064, 0.178):
        t = t0 + r; cells.append(f'router {r:.3f}: {t:.2f} ms, bar {ap - F(t) - 0.003:+.4f}, env+0.003 {ap - env(t) - 0.003:+.4f}')
    print(f'{m.group(1):18s} {[(i, round(s, 2)) for i, s in rungs]} AP {ap:.4f} null+0.003 {ap - nl - 0.003:+.4f} | ' + ' | '.join(cells))
