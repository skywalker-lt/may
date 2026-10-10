# Per-cluster headroom in model space, BUDGETED (share-matched), out-of-fold.
# For M/L: on fold A rank scene clusters by their L-M AP gain, send whole clusters to L in that order until the L share s
# is reached (the last cluster split at random), apply the same cluster ranking to fold B; score full val2017 mixture;
# share-matched null = random route at the realised share. Also: per-cluster gain spread vs its bootstrap noise.
import numpy as np
from common import *
M=load('dumpml_yolo26m_coco'); L=load('dump_yolo26l_coco'); M12=load('dump_yolov12m_sdpa_coco')
F=feats(M); n=len(M['imgIds']); Y=gt_matrix(M); cnt=Y.sum(1)
def budget_route(recs,K,fk,share,seeds=range(3),key=None):
    out=[];nul=[]
    for sd in seeds:
        rng=np.random.default_rng(500+sd); perm=rng.permutation(n); folds=[perm[:n//2],perm[n//2:]]
        A=np.zeros(n,int)
        for f in range(2):
            tr,te=folds[f],folds[1-f]
            X=F[fk] if key is None else key
            c,lab=kmeans(X[tr],K,rng); labte=assign(X[te],c)
            g=[]
            for k in range(K):
                m=np.zeros(n,bool); m[tr[lab==k]]=True
                g.append(ap_rec(recs[1],m)-ap_rec(recs[0],m) if m.sum()>20 else -1)
            order=np.argsort(-np.array(g)); need=int(round(share*len(te))); chosen=[]
            for k in order:
                idx=te[labte==k]
                if need<=0: break
                take=idx if len(idx)<=need else rng.choice(idx,need,replace=False)
                chosen.append(take); need-=len(take)
            if chosen: A[np.concatenate(chosen)]=1
        out.append(ap_mix(recs,A)); nul.append(ap_mix(recs,rng.permutation(A)))
    return np.mean(out),np.mean(nul),np.std(np.array(out)-np.array(nul))/np.sqrt(len(out))
print('M %.4f L %.4f 12m %.4f'%(ap_rec(M),ap_rec(L),ap_rec(M12)),flush=True)
for share in [0.16,0.31,0.42]:
    for fk in ['m640','n320']:
        for K in [4,16]:
            a,nu,se=budget_route([M,L],K,fk,share)
            print(f'M/L share {share:.2f} {fk} K={K:2d}: scene-cluster route {a:.4f}  null {nu:.4f}  diff {a-nu:+.4f} (se {se:.4f})',flush=True)
# family choice by scene: equal-share 12m (budget 0.5) and unconstrained
for fk in ['m640','n320']:
    for K in [8]:
        a,nu,se=budget_route([M,M12],K,fk,0.5)
        print(f'M/12m share 0.50 {fk} K={K:2d}: scene-cluster route {a:.4f}  null {nu:.4f}  diff {a-nu:+.4f} (se {se:.4f})',flush=True)
