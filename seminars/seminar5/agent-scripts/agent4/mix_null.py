import numpy as np, sys
sys.argv=[sys.argv[0]]
exec(open('/data/tmp/ds-yolo/seminar5/work/agent4/mix.py').read().split('rng=np.random.default_rng(1)')[0])
pl_m=np.load(W+'pred_large_m640.npy')
for seed in (11,12,13): ev(f'null random 0.5 seed {seed}',top(np.random.default_rng(seed).random(len(ids)),0.5))
for seed in (21,22): ev(f'null random 0.3 seed {seed}',top(np.random.default_rng(seed).random(len(ids)),0.3))
ev('learned m640 large-count 0.3',top(pl_m,0.3))
