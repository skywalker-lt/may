"""Agent 9, round 2. Re-score-only bounds (teacher score moved onto the base's own boxes at IoU>=0.5, same class; routed
boxes with no teacher match demoted x0.1; nothing added). ONE thread. Rows:
  L512 base: X re-score in 16 random tiles (null of the re-score form); X re-score on all 100 tiles (dense re-score
             bound); X re-score on whole images, the 16% of images with the largest uncertainty mass (per-image form at
             the same average cost).
  L544 base: X and L640 re-score in the 16 uncertainty tiles (tile unit); X re-score on the k=64 queries with the largest
             s(1-s) per image, and on 64 random queries (agent 1's unit and its strict null); L640 on the k=64 queries."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np
part=sys.argv[1] if len(sys.argv)>1 else 'all'; sys.argv=sys.argv[:1]   # keep teacher_tiles' module body idle on import
sys.path.insert(0,'/data/tmp/ds-yolo/seminar6/work/agent9')
import teacher_tiles as T
from teacher_tiles import load, run, unc_route, rand_route, anchored, tile_of, iou, N, G
T.SRC['L544']='/data/tmp/ds-yolo/seminar6/inputs/dumps_r2/dumpml_yolo26l_544_coco.json'
def rescore(B,E,inB,inE):
    """base rows in inB take the score of their greedy same-class IoU>=0.5 match among teacher rows in inE; unmatched x0.1"""
    Bi=B[:,0].astype(int); Ei=E[:,0].astype(int)
    a=np.where(inB)[0]; b=np.where(inE)[0]; newscore=B[:,5].copy(); matched=np.zeros(len(B),bool)
    ao=a[np.argsort(Bi[a],kind='stable')]; bo=b[np.argsort(Ei[b],kind='stable')]
    ab=np.searchsorted(Bi[ao],np.arange(N+1)); bb=np.searchsorted(Ei[bo],np.arange(N+1))
    for i in range(N):
        A=ao[ab[i]:ab[i+1]]; Q=bo[bb[i]:bb[i+1]]
        if len(A)==0 or len(Q)==0: continue
        U=iou(B[A,1:5],E[Q,1:5])*(B[A,6][:,None]==E[Q,6][None,:])
        for q in np.argsort(-E[Q,5]):
            c=U[:,q].argmax()
            if U[c,q]>=0.5: newscore[A[c]]=E[Q[q],5]; matched[A[c]]=True; U[c,:]=-1
    R=B.copy(); R[:,5]=newscore; R[inB&~matched,5]*=0.1; return R
def query_route(B,k,rng,random=False):
    """per image, the k detections with the largest s(1-s) (or k at random); returns a bool mask over B's rows"""
    Bi=B[:,0].astype(int); key=B[:,5]*(1-B[:,5]) if not random else rng.random(len(B))
    order=np.lexsort((-key,Bi)); rank=np.empty(len(B),int)
    first=np.searchsorted(Bi[order],np.arange(N+1))
    for i in range(N): rank[order[first[i]:first[i+1]]]=np.arange(first[i+1]-first[i])
    return rank<k
rng=np.random.default_rng(0)
if part in ('A','all'):
    B=load('L512'); X=load('X'); Bi=B[:,0].astype(int); tb=tile_of(B,G); Xi=X[:,0].astype(int); tx=anchored(B,X,G,'L512X')
    S=unc_route(B,G,16,rng); Sr=rand_route(G,16,rng)
    run('A1 L512: X re-score only, 16 random tiles (null of the re-score form)',rescore(B,X,Sr[Bi,tb],Sr[Xi,tx]))
    run('A2 L512: X re-score only, all 100 tiles (dense re-score bound)',rescore(B,X,np.ones(len(B),bool),np.ones(len(X),bool)))
    s=B[:,5]>=0.01; m=np.zeros(N); np.add.at(m,Bi[s],B[s,5]*(1-B[s,5])); top=np.zeros(N,bool); top[np.argsort(-m)[:int(round(0.16*N))]]=True
    run('A3 L512: X re-score only, whole images, top 16% by uncertainty mass (per-image form)',rescore(B,X,top[Bi],top[Xi]))
if part in ('B','all'):
    B=load('L544'); X=load('X'); L=load('L'); Bi=B[:,0].astype(int); tb=tile_of(B,G)
    run('B0 L544 alone (check: 0.5329)',B)
    S=unc_route(B,G,16,rng)
    for n,E in (('X',X),('L640',L)):
        Ei=E[:,0].astype(int); te=anchored(B,E,G,'L544'+('X' if n=='X' else 'L'))
        run(f'B1 L544: {n} re-score only, 16 uncertainty tiles',rescore(B,E,S[Bi,tb],S[Ei,te]))
    q=query_route(B,64,rng); qr=query_route(B,64,rng,random=True)
    print(f'   candidates per image: mean {len(B)/N:.1f}; routed share of candidates at k=64: {q.mean():.3f}',flush=True)
    allX=np.ones(len(X),bool); allL=np.ones(len(L),bool)
    run('B2 L544: X re-score only, k=64 queries by s(1-s)',rescore(B,X,q,allX))
    run('B3 L544: X re-score only, 64 random queries (strict null)',rescore(B,X,qr,allX))
    run('B4 L544: L640 re-score only, k=64 queries by s(1-s)',rescore(B,L,q,allL))
print('DONE2')
