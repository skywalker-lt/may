"""Which teacher a per-tile GT oracle picks inside the routed tiles (strict argmax among tiles where some teacher gains)."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar6/work/agent9')
from teacher_tiles import load, quality, unc_route, tile_gain, gt, gi, N, tile_of, G
for base,names in (('M',('L','X','M768')),('L512',('L','X'))):
    B=load(base); rng=np.random.default_rng(0); S=unc_route(B,G,16,rng); qb=quality(B)
    g={n:tile_gain(qb,quality(load(n)),G) for n in names}
    Gs=np.stack([g[n] for n in names],-1); best=Gs.argmax(-1); mx=Gs.max(-1)
    pos=S&(mx>1e-9); neg=S&(mx<=1e-9)
    shares=[(best[pos]==j).mean() for j in range(len(names))]
    print(f'base {base}: routed tiles {S.sum()}; with a gaining teacher {pos.sum()} ({pos.sum()/S.sum():.2f}); strict-argmax shares '+' / '.join(f'{n} {s:.2f}' for n,s in zip(names,shares)),flush=True)
    for j,n in enumerate(names):
        others=np.delete(Gs,j,-1).max(-1)
        print(f'   {n}: total gain in routed tiles {g[n][S].sum():.1f}; gain where it is the oracle choice {Gs[...,j][pos&(best==j)].sum():.1f}; '
              f'mean margin over the best other teacher where chosen {(Gs[...,j]-others)[pos&(best==j)].mean():.3f}; oracle total {mx[pos].sum():.1f}',flush=True)
