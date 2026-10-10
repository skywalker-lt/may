"""Agent 8, round 2. The dense re-scorer null for lens 9's RTR: X@640's scores on L@512's own boxes (agent 9's re-score-only
rule: matched base dets take the teacher's score, unmatched base dets inside the selected tiles x0.1, nothing added) applied
in (a) all 100 tiles (a dense cheap re-scoring head's ceiling), (b) 16 random tiles (the re-score tile null), against agent 9's
16 uncertainty tiles (0.5440). Reads agent 9's caches read-only. ONE thread."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar6/work/agent9')
from teacher_tiles import load, run, unc_route, rand_route, anchored, tile_of, iou, N, G
B=load('L512'); Bi=B[:,0].astype(int); tb=tile_of(B,G); rng=np.random.default_rng(0)
E=load('X'); Ei=E[:,0].astype(int); te=anchored(B,E,G,'L512X')
def rescore(S,tag):
    inB=S[Bi,tb]; inE=S[Ei,te]; a=np.where(inB)[0]; b=np.where(inE)[0]
    newscore=B[:,5].copy(); matched=np.zeros(len(B),bool)
    ao=a[np.argsort(Bi[a],kind='stable')]; bo=b[np.argsort(Ei[b],kind='stable')]
    ab=np.searchsorted(Bi[ao],np.arange(N+1)); bb=np.searchsorted(Ei[bo],np.arange(N+1))
    for i in range(N):
        A=ao[ab[i]:ab[i+1]]; Bq=bo[bb[i]:bb[i+1]]
        if len(A)==0 or len(Bq)==0: continue
        U=iou(B[A,1:5],E[Bq,1:5])*(B[A,6][:,None]==E[Bq,6][None,:])
        for q in np.argsort(-E[Bq,5]):
            c=U[:,q].argmax()
            if U[c,q]>=0.5: newscore[A[c]]=E[Bq[q],5]; matched[A[c]]=True; U[c,:]=-1
    R2=B.copy(); R2[:,5]=newscore; R2[inB&~matched,5]*=0.1
    print(f'{tag}: selected tiles/image {S.sum(1).mean():.1f}, base dets in them {inB.sum()}, matched {matched[inB].mean():.3f}',flush=True)
    run(tag,R2)
rescore(np.ones((N,G*G),bool),'X re-score only, ALL 100 tiles of L@512 (dense re-scorer ceiling)')
rescore(rand_route(G,16,rng),'X re-score only, 16 RANDOM tiles of L@512 (re-score tile null)')
rescore(unc_route(B,G,8,rng),'X re-score only, 8 uncertainty tiles of L@512')
print('DONE')
