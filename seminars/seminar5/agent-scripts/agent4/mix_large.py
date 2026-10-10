import numpy as np, sys
from scipy.stats import spearmanr
sys.argv=[sys.argv[0]]
exec(open('/data/tmp/ds-yolo/seminar5/work/agent4/mix.py').read().split('rng=np.random.default_rng(1)')[0])
n320=z['n320'][[zi[int(i)] for i in ids]]
yl=np.log1p(f['large'])
for xn,X in [('n320',n320),('m640',m640)]:
    p=ridge_oof(X,yl); print(f'large-count predictability {xn}: spearman {spearmanr(p,f["large"]).correlation:+.3f}',flush=True); np.save(W+f'pred_large_{xn}.npy',p)
# area-weighted: fraction of image covered by GT boxes
pl_n=np.load(W+'pred_large_n320.npy'); pl_m=np.load(W+'pred_large_m640.npy')
rng=np.random.default_rng(7)
ev('learned n320 large-count 0.5',top(pl_n,0.5))
ev('learned m640 large-count 0.5',top(pl_m,0.5))
ev('null random 0.3',top(rng.random(len(ids)),0.3))
ev('GT large-object count rule 0.3',top(f['large']+1e-3*rng.random(len(ids)),0.3))
ev('learned n320 large-count 0.3',top(pl_n,0.3))
