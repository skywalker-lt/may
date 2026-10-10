"""Round 2, attack on lens 9: split X@640's re-score-only gain on L@512's routed boxes (agent 9: 0.5440, +0.0178) into
(i) per-image calibration (X's score multiset per image, assigned in L's own within-image order) and (ii) within-image
re-ranking (the remainder). Uses agent 9's helpers read-only. ONE thread."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar6/work/agent9')
from teacher_tiles import load, run, unc_route, anchored, tile_of, iou, N, G
B=load('L512'); Bi=B[:,0].astype(int); tb=tile_of(B,G); rng=np.random.default_rng(0); S=unc_route(B,G,16,rng); inB=S[Bi,tb]
E=load('X'); Ei=E[:,0].astype(int); te=anchored(B,E,G,'L512X'); inE=S[Ei,te]
a=np.where(inB)[0]; b=np.where(inE)[0]
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
full=newscore.copy(); full[inB&~matched]*=0.1
# (i) calibration only: within each image, the routed dets keep L's order but take the sorted multiset of 'full' scores
cal=B[:,5].copy()
for i in range(N):
    A=ao[ab[i]:ab[i+1]]
    if len(A)==0: continue
    order=np.argsort(-B[A,5]); cal[A[order]]=np.sort(full[A])[::-1]
# (iii) image-level temperature only: one scalar per image, the ratio of mean 'full' to mean L score on routed dets, applied to routed dets
tmp=B[:,5].copy()
for i in range(N):
    A=ao[ab[i]:ab[i+1]]
    if len(A)==0: continue
    tmp[A]=np.clip(B[A,5]*(full[A].mean()/max(B[A,5].mean(),1e-6)),0,1)
R=B.copy(); R[:,5]=cal; run('X re-score, L order kept (per-image calibration only)',R)
R=B.copy(); R[:,5]=tmp; run('X re-score, one scalar per image (image temperature only)',R)
print('DONE')
