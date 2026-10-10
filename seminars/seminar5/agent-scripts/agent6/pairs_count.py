# Realisable count routing (out-of-fold ridge on the n320 thumbnail / m640 pooled feature -> log GT count) for same-cost pairs.
import sys, json, numpy as np
from mixacc import Mix
from scipy.stats import spearmanr
A,B,tag=sys.argv[1],sys.argv[2],sys.argv[3]
M=Mix(A,B); nI=len(M.imgIds)
INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
z=np.load(INP+'val2017_stem_pooled.npz'); order=np.array([M.img_index[i] for i in z['image_id']])
N320=np.zeros((nI,128)); N320[order]=z['n320']; M640=np.zeros((nI,512)); M640[order]=z['m640']
gt=json.load(open(INP+'instances_val2017.json')); cnt=np.zeros(nI)
for a in gt['annotations']:
    if not a.get('iscrowd',0): cnt[M.img_index[a['image_id']]]+=1
y=np.log1p(cnt); rng=np.random.RandomState(0); folds=rng.permutation(nI)%5
def ridge_oof(X,y,lam=1.0):
    p=np.zeros(nI)
    for f in range(5):
        tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd; A_=Xs[tr]
        w=np.linalg.solve(A_.T@A_+lam*len(A_)*np.eye(X.shape[1])/100, A_.T@(y[tr]-y[tr].mean())); p[~tr]=Xs[~tr]@w+y[tr].mean()
    return p
g=np.load(f'gain_{tag}.npy')
for nm,X in (('n320',N320),('m640',M640)):
    p=ridge_oof(X,y); print(tag, nm, 'rho(pred count, GT count)', round(spearmanr(p,cnt).correlation,3), 'rho(pred count, gain)', round(spearmanr(p,g).correlation,3))
    for share in (0.25,0.5):
        n=int(share*nI); s=np.zeros(nI,np.int8); s[np.argsort(p)[:n]]=1
        nulls=[]
        for d in range(5):
            r=np.zeros(nI,np.int8); r[rng.permutation(nI)[:n]]=1; nulls.append(M.ap(r))
        print(f'  share {share} sparse->B learned-count {M.ap(s):.4f}  null mean {np.mean(nulls):.4f} sd {np.std(nulls):.4f}', flush=True)
