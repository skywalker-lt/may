# Headroom of scene-routed class priors ("label-space mixture of domains") on YOLO26-M's val2017 dump.
# score' = score * exp(b[cluster(img), class]); b fit on fold A (GT presence per cluster), applied to fold B; 2-fold, repeated.
import sys, numpy as np
from common import *
r=load('dumpml_yolo26m_coco'); Y=gt_matrix(r); P=(Y>0).astype(float); F=feats(r); n=len(P)
base=ap_rec(r); print('base M AP %.4f'%base)
lsc=np.log(r['score'])
def rescore(B):  # B: (nimg,80) log-bias
    return np.exp(lsc+B[r['img'],r['cat']])
# 1. oracle: GT class presence
for eps in [1e-1,1e-2,1e-3]:
    B=np.where(P>0,0.0,np.log(eps)); print('oracle GT presence, absent x%g: %.4f'%(eps,ap_rec(r,score=rescore(B))))
def prior_bias(lab_tr,P_tr,lab_te,K,m=5.0):
    pc=P_tr.mean(0)+1e-4
    q=np.zeros((K,80))
    for k in range(K):
        s=lab_tr==k; q[k]=(P_tr[s].sum(0)+m*pc)/(s.sum()+m)
    return np.log(q/pc)[lab_te]
def knn_bias(X_tr,P_tr,X_te,k=50,m=5.0):
    pc=P_tr.mean(0)+1e-4
    Xn_tr=X_tr/np.linalg.norm(X_tr,axis=1,keepdims=True); Xn_te=X_te/np.linalg.norm(X_te,axis=1,keepdims=True)
    S=Xn_te@Xn_tr.T; idx=np.argpartition(-S,k,axis=1)[:,:k]
    q=(P_tr[idx].sum(1)+m*pc)/(k+m); return np.log(q/pc)
rows=[]
seeds=[0,1,2]
alphas=[0.5,1.0]
def run(name,fn):
    res={a:[] for a in alphas}
    for sd in seeds:
        rng=np.random.default_rng(100+sd); perm=rng.permutation(n); folds=[perm[:n//2],perm[n//2:]]
        Bfull=np.zeros((n,80))
        for f in range(2):
            tr,te=folds[f],folds[1-f]
            Bfull[te]=fn(tr,te,rng)
        for a in alphas: res[a].append(ap_rec(r,score=rescore(a*Bfull)))
    line=name+'  '+'  '.join('a=%.2f: %.4f (+%.4f, sd %.4f)'%(a,np.mean(v),np.mean(v)-base,np.std(v)) for a,v in res.items())
    print(line,flush=True)
for K in [4,8,16,32,64]:
    for fk in ['m640','n320']:
        X=F[fk]
        def fn(tr,te,rng,X=X,K=K):
            c,lab=kmeans(X[tr],K,rng); return prior_bias(lab,P[tr],assign(X[te],c),K)
        run(f'scene K={K:2d} router={fk}',fn)
    def fnull(tr,te,rng,K=K):
        lab=rng.integers(K,size=len(tr)); return prior_bias(lab,P[tr],rng.integers(K,size=len(te)),K)
    run(f'null  K={K:2d} random   ',fnull)
    # GT partition bound: clusters from GT presence on train fold, test images assigned by their own GT presence (oracle route)
    def fgt(tr,te,rng,K=K):
        c,lab=kmeans(P[tr],K,rng); return prior_bias(lab,P[tr],assign(P[te],c),K)
    run(f'GT-partition K={K:2d} (oracle route)',fgt)
for fk in ['m640','n320']:
    for k in [25,100]:
        X=F[fk]; run(f'dense kNN k={k} {fk}',lambda tr,te,rng,X=X,k=k: knn_bias(X[tr],P[tr],X[te],k))
