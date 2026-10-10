"""Calibration-free routing signal: per-GT quality q = best same-class det score with IoU>=t (0 if none), for M and E.
Net per-object gain of E over M, and the share of it that falls in the tiles a GT-free router picks at share s."""
import sys
from mix_eval import *
from anchor_tiles import iou
def quality(A,thr):
    q=np.zeros(len(gt)); Ai=A[:,0].astype(int); gi=gt[:,0].astype(int)
    ao=np.argsort(Ai,kind='stable'); go=np.argsort(gi,kind='stable')
    ab=np.searchsorted(Ai[ao],np.arange(N+1)); gb=np.searchsorted(gi[go],np.arange(N+1))
    for i in range(N):
        a=ao[ab[i]:ab[i+1]]; g=go[gb[i]:gb[i+1]]
        if len(a)==0 or len(g)==0: continue
        U=iou(gt[g,1:5],A[a,1:5])*(gt[g,6][:,None]==A[a,6][None,:])
        q[g]=np.where(U>=thr,A[a,5][None,:],0).max(1)
    return q
M=load(MODELS['M']); G=10; rng=np.random.default_rng(0); tg=tile_of(gt,G); gi=gt[:,0].astype(int)
small=gt[:,5]<32**2
RULES=('Mdet0.01','Munc0.01','Mdet0.3','random')
for ename in sys.argv[1:]:
    E=load(MODELS[ename])
    for thr in (0.5,):
        qm=quality(M,thr); qe=quality(E,thr); d=qe-qm
        print(f'{ename} IoU>={thr}: mean q M {qm.mean():.4f} E {qe.mean():.4f}; net gain sum {d.sum():.1f}; positive {d[d>0].sum():.1f} negative {d[d<0].sum():.1f}; small-object share of net {d[small].sum()/d.sum():.2f}')
        # gain by M's own quality bin
        bins=[0,1e-9,0.1,0.3,0.5,0.7,1.01]; r=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            m=(qm>=lo)&(qm<hi); r.append(f'[{lo:.1f},{hi:.1f}) n={m.sum()} mean d={d[m].mean():+.3f}')
        print('   by M quality:',' | '.join(r))
        for rule in RULES:
            sc=scores(rule,G,M,E,rng); out=[]
            for s in (0.10,0.16,0.25,0.40):
                k=int(round(s*100)); sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,100),bool); np.put_along_axis(S,sel,True,1)
                ins=S[gi,tg]; out.append(f's={s:.2f}: {d[ins].sum()/d.sum():.3f}')
            print(f'   share of net gain in {rule} tiles:',' '.join(out))
