"""Round 3: (1) size-boundary diagnosis of the P3-only proxy (agent 4/6 rule: each detection sized by itself; agent 5 rule:
one size per object, from M's match); (2) re-score vs geometry decomposition (agent 9's method) restricted to the anchors
TCR re-predicts, for E in {L, M768, X}; AP, AP50, AP75. Uncertainty-mass router, share 0.16, 10x10 tiles, object-anchored."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, numpy as np, contextlib, io
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent5'); sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent5/r2')
from tiles_common import load, tile_of, N, R, ids
from mix_eval import scores, MODELS, cg
from anchor_tiles import anchored_tiles, iou
from pycocotools.cocoeval import COCOeval
def ev(a):
    with contextlib.redirect_stdout(io.StringIO()):
        cd=cg.loadRes(np.column_stack([np.array(ids)[a[:,0].astype(int)],a[:,1:7]]))
        e=COCOeval(cg,cd,'bbox'); e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[[0,1,2,3,4,5]]
def size(a): i=a[:,0].astype(int); return np.sqrt(a[:,3]*a[:,4])*R[i]
def run(tag,a):
    s=ev(a); print(f'{tag:70s} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S/M/L {s[3]:.4f}/{s[4]:.4f}/{s[5]:.4f}',flush=True)
M=load(MODELS['M']); G=10; k=16; rng=np.random.default_rng(0)
Mi=M[:,0].astype(int); tm=tile_of(M,G); sm=size(M)
run('M alone',M)
for en in sys.argv[1].split(','):
    E=load(MODELS[en]); Ei=E[:,0].astype(int)
    sc=scores('Munc0.01',G,M,E,rng); sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1)
    te=anchored_tiles(M,E,G,'M'+en); inM=S[Mi,tm]; inE=S[Ei,te]
    mm=np.load(f'/data/tmp/ds-yolo/seminar5/work/agent5/r2/cache_match_M{en}.npy') if os.path.exists(f'/data/tmp/ds-yolo/seminar5/work/agent5/r2/cache_match_M{en}.npy') else None
    se_own=size(E); se_obj=se_own.copy()
    if mm is not None: ok=mm>=0; se_obj[ok]=sm[mm[ok]]
    for thr in [float(t) for t in sys.argv[2].split(',')]:
        # boundary diagnosis
        if en=='L' and thr==64:
            ok=mm>=0
            lost=((sm[mm[ok]]<thr)&(se_own[ok]>=thr)&inE[ok]).sum(); dup=((sm[mm[ok]]>=thr)&(se_own[ok]<thr)&inE[ok]).sum()
            print(f'BOUNDARY thr=64: matched L dets in routed tiles whose own size and M-match size straddle 64 px: M<64,L>=64 (object lost under self-sizing) {lost}; M>=64,L<64 (duplicated) {dup}',flush=True)
            run(f'{en} <{thr:.0f} self-sized (agent 4/6 rule)', np.concatenate([M[~(inM&(sm<thr))], E[inE&(se_own<thr)]]))
        outM=inM&(sm<thr); inEs=inE&(se_obj<thr)
        run(f'{en} <{thr:.0f} object-sized, full replacement', np.concatenate([M[~outM],E[inEs]]))
        # re-score + add with M's geometry (agent 9), restricted to the same objects
        a=np.where(outM)[0]; b=np.where(inEs)[0]
        newscore=M[:,5].copy(); matched=np.zeros(len(M),bool); Em=np.zeros(len(E),bool)
        ao=a[np.argsort(Mi[a],kind='stable')]; bo=b[np.argsort(Ei[b],kind='stable')]
        ab=np.searchsorted(Mi[ao],np.arange(N+1)); bb=np.searchsorted(Ei[bo],np.arange(N+1))
        for i in range(N):
            A=ao[ab[i]:ab[i+1]]; B=bo[bb[i]:bb[i+1]]
            if len(A)==0 or len(B)==0: continue
            U=iou(M[A,1:5],E[B,1:5])*(M[A,6][:,None]==E[B,6][None,:])
            for q in np.argsort(-E[B,5]):
                c=U[:,q].argmax()
                if U[c,q]>=0.5: newscore[A[c]]=E[B[q],5]; matched[A[c]]=True; Em[B[q]]=True; U[c,:]=-1
        R2=M.copy(); R2[:,5]=newscore; R2[outM&~matched,5]*=0.1
        run(f'{en} <{thr:.0f} re-score + add (M geometry)', np.concatenate([R2,E[inEs&~Em]]))
print('DONE')
