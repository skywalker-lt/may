import sys, numpy as np, pickle
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent10')
from common import *
from scipy.stats import spearmanr
G=gt(); z=np.load(D+'/val2017_stem_pooled.npz')
ids=z['image_id']; X=z['n320'].astype(np.float64)
cnt=np.zeros(len(ids)); sm=np.zeros(len(ids))
for i,img in enumerate(ids):
    a=[g for g in G.imgToAnns[int(img)] if not g['iscrowd']]
    cnt[i]=len(a); sm[i]=sum(g['area']<32**2 for g in a)
def oof_ridge(X,y,lam=10.0,k=5,seed=0):
    rng=np.random.default_rng(seed); f=rng.permutation(len(y))%k; p=np.zeros(len(y))
    for j in range(k):
        tr=f!=j; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6
        A=(X[tr]-mu)/sd; b=y[tr]-y[tr].mean()
        w=np.linalg.solve(A.T@A+lam*len(A)*np.eye(A.shape[1])/100,A.T@b)
        p[~tr]=((X[~tr]-mu)/sd)@w+y[tr].mean()
    return p
pc=oof_ridge(X,np.log1p(cnt)); ps=oof_ridge(X,np.log1p(sm))
print('spearman n320-pred vs GT count %.3f, small count %.3f'%(spearmanr(pc,cnt)[0],spearmanr(ps,sm)[0]))
pickle.dump({'ids':ids,'cnt':cnt,'small':sm,'pred_cnt':pc,'pred_small':ps},open(W+'/features.pkl','wb'))
