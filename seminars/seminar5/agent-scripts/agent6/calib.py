# Per-image routed score temperature/shift (one scalar or a per-image affine on all logits): is there learnable headroom?
import json, numpy as np, pickle
from fastacc import FastAcc
INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
F=FastAcc('evm.pkl'); nI=len(F.imgIds); nK=len(F.catIds)
D=pickle.load(open('evm.pkl','rb')); nA=len(D['areaRng'])
# per-image calibration target: over dets with score>0.05, mean(match at IoU .5:.95) - mean(score)
num=np.zeros(nI); den=np.zeros(nI); ms=np.zeros(nI); cnt_det=np.zeros(nI)
for k in range(nK):
    for i in range(nI):
        e=D['recs'][k*nA*nI+i]
        if e is None: continue
        s=e[2][:100]; m=(e[3][:,:100]>0).mean(0); ig=e[4][:,:100].mean(0)>0.5
        w=(s>0.05)&~ig
        num[F.img_index[e[0]]]+=np.sum(m[w]); ms[F.img_index[e[0]]]+=np.sum(s[w]); den[F.img_index[e[0]]]+=w.sum()
tgt=np.where(den>0,(num-ms)/np.maximum(den,1),0)
z=np.load(INP+'val2017_stem_pooled.npz'); order=np.array([F.img_index[i] for i in z['image_id']])
N320=np.zeros((nI,128)); N320[order]=z['n320']; M640=np.zeros((nI,512)); M640[order]=z['m640']
own=np.stack([den, ms/np.maximum(den,1), np.log1p(den)],1)
rng=np.random.RandomState(0); folds=rng.permutation(nI)%5
def ridge_oof(X,y,lam=1.0):
    p=np.zeros(nI)
    for f in range(5):
        tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd
        A=Xs[tr]; w=np.linalg.solve(A.T@A+lam*len(A)*np.eye(X.shape[1]), A.T@(y[tr]-y[tr].mean()))
        p[~tr]=Xs[~tr]@w+y[tr].mean()
    return p
print('base', round(F.ap(),4))
# oracle: shift each image's logits by c*target (GT-derived)
for c in (1,2,4,8):
    print('oracle per-image shift c=%g'%c, round(F.ap(logit_add=np.repeat((c*tgt)[:,None],nK,1)),4))
from scipy.stats import spearmanr
for nm,X in (('n320',N320),('m640',M640),('own-stats',own),('m640+own',np.hstack([M640,own]))):
    p=ridge_oof(X,tgt); r=spearmanr(p,tgt).correlation
    out=[(c,F.ap(logit_add=np.repeat((c*(p-p.mean()))[:,None],nK,1))) for c in (1,2,4,8)]
    nul=F.ap(logit_add=np.repeat((4*(p-p.mean()))[rng.permutation(nI)][:,None],nK,1))
    print(f'learned {nm}: rho={r:.3f} '+' '.join(f'c{c}:{a:.4f}' for c,a in out)+f' perm-null c4:{nul:.4f}')
