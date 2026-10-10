# Joint P(A >= bar AND delta >= 0.003), and dense control alone at A's bars (same split-normal reading as probs.py).
import numpy as np
rng = np.random.default_rng(1); N = 400000; z = 1.2816
def sn(lo, c, hi):
    u = rng.standard_normal(N); return np.where(u < 0, c + u*(c-lo)/z, c + u*(hi-c)/z)
for reg, C in (("80ep", (0.516, 0.522, 0.527)), ("600ep", (0.506, 0.515, 0.523))):
    c = sn(*C)
    print(reg, "dense control alone: P(>=0.5273)=%.3f P(>=0.5288)=%.3f" % (np.mean(c >= .5273), np.mean(c >= .5288)))
    for nm, d in (("A-tied", (-0.006, -0.001, 0.003) if reg == "80ep" else (-0.008, 0.0, 0.006)),
                  ("A6", (-0.004, 0.0, 0.004) if reg == "80ep" else (-0.006, 0.001, 0.007))):
        dd = sn(*d); a = c + dd
        print(f"  {nm}: joint P(>=0.5273 & d>=.003)={np.mean((a>=.5273)&(dd>=.003)):.3f}  joint P(>=0.5288 & d>=.003)={np.mean((a>=.5288)&(dd>=.003)):.3f}")
