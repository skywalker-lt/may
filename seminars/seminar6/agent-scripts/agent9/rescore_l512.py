"""Re-score vs geometry split of the teacher's in-tile gain on the L@512 base (agent 5's round-3 method, decomp_r3.py),
uncertainty route of L@512, 16 of 100 tiles, every anchor. Variants: re-score only (base geometry, unmatched base dets
demoted x0.1, nothing added), re-score + add (teacher's unmatched dets added). ONE thread."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar6/work/agent9')
from teacher_tiles import load, run, unc_route, anchored, tile_of, iou, N, G
B=load('L512'); Bi=B[:,0].astype(int); tb=tile_of(B,G); rng=np.random.default_rng(0); S=unc_route(B,G,16,rng); inB=S[Bi,tb]
for en in sys.argv[1].split(','):
    E=load(en); Ei=E[:,0].astype(int); te=anchored(B,E,G,'L512'+en); inE=S[Ei,te]
    a=np.where(inB)[0]; b=np.where(inE)[0]
    newscore=B[:,5].copy(); matched=np.zeros(len(B),bool); Em=np.zeros(len(E),bool)
    ao=a[np.argsort(Bi[a],kind='stable')]; bo=b[np.argsort(Ei[b],kind='stable')]
    ab=np.searchsorted(Bi[ao],np.arange(N+1)); bb=np.searchsorted(Ei[bo],np.arange(N+1))
    for i in range(N):
        A=ao[ab[i]:ab[i+1]]; Bq=bo[bb[i]:bb[i+1]]
        if len(A)==0 or len(Bq)==0: continue
        U=iou(B[A,1:5],E[Bq,1:5])*(B[A,6][:,None]==E[Bq,6][None,:])
        for q in np.argsort(-E[Bq,5]):
            c=U[:,q].argmax()
            if U[c,q]>=0.5: newscore[A[c]]=E[Bq[q],5]; matched[A[c]]=True; Em[Bq[q]]=True; U[c,:]=-1
    R2=B.copy(); R2[:,5]=newscore; R2[inB&~matched,5]*=0.1
    print(f'{en}: routed base dets {inB.sum()}, matched to the teacher {matched[inB].mean():.3f}; teacher dets in routed tiles {inE.sum()}, unmatched {(~Em[inE]).mean():.3f}',flush=True)
    if en=='X': run(f'{en}@640 re-score only on L@512 geometry (no boxes added)',R2)
    run(f'{en}@640 re-score + add on L@512 geometry',np.concatenate([R2,E[inE&~Em]]))
print('DONE')
