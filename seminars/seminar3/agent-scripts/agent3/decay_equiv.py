"""Agent 3, round 3: SGD with decay on four full kernels equals base + zero-sum deviations with the same decay on both."""
import numpy as np
rng = np.random.default_rng(1); lr, wd = 0.01, 5e-4
B = rng.standard_normal(50); D = rng.standard_normal((4, 50)) * 0.01; D -= D.mean(0); W = B + D
B0, D0 = B.copy(), D.copy()
for _ in range(1000):
    g = rng.standard_normal((4, 50)) * 0.0  # zero task gradient: isolate the decay
    W = W * (1 - lr * wd) - lr * g
print('full kernels, decay only: |dev| ratio', np.linalg.norm(W - W.mean(0)) / np.linalg.norm(D0), 'base ratio', np.linalg.norm(W.mean(0)) / np.linalg.norm(B0), '(1-lr*wd)^1000', (1 - lr * wd) ** 1000)
print('sum_e |B+D_e|^2 - (4|B|^2 + sum|D_e|^2)', (np.square(B0 + D0).sum() - 4 * np.square(B0).sum() - np.square(D0).sum()))
# decay half-life in optimiser steps at the plan's upcycle lr0 and the default decay
for lr0 in (0.00038, 0.01): print('lr', lr0, 'half-life of an unsupported deviation (steps)', np.log(2) / (lr0 * 5e-4))
# shrinkage of an unsupported deviation over a run: steps = ceil(118287/256) * epochs; decay 5e-4 scaled 1x or 4x
# (batch/nbs), mean lr factor 0.5 (decaying schedule) or 1.0 (constant). Not verified against the trainer.
for name, steps, lr0 in (('80 ep upcycled', 463 * 80, 0.00038), ('600 ep scratch', 463 * 600, 0.01)):
    print(name, steps, 'steps; fraction lost:', [round(1 - float(np.exp(-steps * lr0 * 5e-4 * k * f)), 4) for k in (1, 4) for f in (0.5, 1.0)])
