"""Check my routed-tile gain share against agent 5's scores('Munc0.01') selection (no pycocotools)."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent5'); sys.path.insert(0,'/data/tmp/ds-yolo/seminar6/work/agent9')
from teacher_tiles import load, quality, unc_route, gt, gi, N, tile_of, G
from mix_eval import scores
M=load('M'); L=load('L'); X=load('X')
qm=quality(M); tg=tile_of(gt,G)
for name,E in (('L',L),('X',X)):
    d=quality(E)-qm
    rng=np.random.default_rng(0); S9=unc_route(M,G,16,rng)
    rng=np.random.default_rng(0); sc=scores('Munc0.01',G,M,E,rng); sel=np.argsort(-sc,1)[:,:16]; S5=np.zeros((N,100),bool); np.put_along_axis(S5,sel,True,1)
    print(f'{name}: share in my route {d[S9[gi,tg]].sum()/d.sum():.3f}; share in agent-5 route {d[S5[gi,tg]].sum()/d.sum():.3f}; routes agree on {(S9==S5).mean():.4f} of tiles',flush=True)
    for s in (0.10,0.16,0.25):
        k=int(round(s*100)); sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,100),bool); np.put_along_axis(S,sel,True,1)
        print(f'   s={s:.2f}: share {d[S[gi,tg]].sum()/d.sum():.3f}',flush=True)
