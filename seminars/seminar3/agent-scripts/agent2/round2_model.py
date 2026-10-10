"""Agent 2, round 2: normal model behind the probability columns. All inputs are my estimates."""
import math
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
p = lambda mu, sd, thr: 1 - Phi((thr - mu) / sd)
NOISE = 0.0021  # sd of the difference of two single-seed runs, seed sigma 0.0015 each (estimate)
rows = [("(i) E wrapped", .5225, .0027, 0, 1e-9, .5276), ("(i) A-tied", .5225, .0027, 0, .002, .5273), ("(i) A-full", .5225, .0027, -.0005, .002, .5273),
        ("(i) A6", .5225, .0027, .0005, .002, .5273), ("(ii) E wrapped", .515, .0049, 0, 1e-9, .5276), ("(ii) A-tied", .515, .0049, .001, .003, .5273),
        ("(ii) A-full", .515, .0049, -.002, .004, .5273), ("(ii) A6", .515, .0049, .0015, .003, .5273)]
for lab, c, sc, d, sd, bar in rows:
    s = math.hypot(sc, sd)
    print(f"{lab:15s} AP {c + d:.4f} sd {s:.4f}  P(literal {bar}) {p(c + d, s, bar):.3f}  P(twin .5288) {p(c + d, s, .5288):.3f}  P(obs delta>=.003) {p(d, math.hypot(sd, NOISE), .003):.3f}")
print("null false-positive rate, single seed:", round(p(0, NOISE, .003), 3), " with two control seeds:", round(p(0, math.sqrt(.0015**2 * 1.5), .003), 3))
