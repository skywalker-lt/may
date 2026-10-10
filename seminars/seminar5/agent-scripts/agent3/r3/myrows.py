import sys; sys.path.insert(0, '.'); from envelope2 import env, F
rows = [('1-bit B16 M640 s0.49', 0.5251, 4.87), ('1-bit B32 M640 s0.49', 0.5187, 4.87),
        ('ladder B16/A/C768 0.49/0.20', 0.5292, 5.13), ('ladder B16/A/C-leaf32 (agent 4)', 0.5276, 5.13),
        ('B16 0.49/A/L 0.30', 0.5318, 5.33), ('exch 512 0.5/A/L 0.45 +router', 0.5308, 5.44),
        ('exch +512-noP3 tau16 0.2 +router', 0.5305, 5.32)]
for d, y, t in rows: print(f'{d:36s} {y:.4f} {t:.2f} bar {y-F(t)-0.003:+.4f} env {env(t):.4f} env+0.003 {y-env(t)-0.003:+.4f}')
# floors at 27% of the way tau16 -> tau32 (full-val one-to-many measurement at share 0.49)
for tag, a16, a32, nl, t in [('L512 s0.49', 0.5239, 0.5141, 0.5155, 4.86*(1-0.179*0.49)), ('L576 s0.49', 0.5353, 0.5273, 0.5278, 5.818*(1-0.179*0.49)),
                             ('L512 s0.40', 0.5250, 0.5187, 0.5173, 4.86*(1-0.179*0.40))]:
    fl = a16 - 0.27 * (a16 - a32)
    print(f'{tag} floor {fl:.4f} at {t:.2f}: bar {fl-F(t)-0.003:+.4f} null+0.003 {fl-nl-0.003:+.4f} env+0.003 {fl-env(t)-0.003:+.4f}; break-even AP {env(t)+0.003:.4f} = loss {0:.0f}'.replace(' = loss 0', f' (fraction {(a16-env(t)-0.003)/(a16-a32):.2f})'))
