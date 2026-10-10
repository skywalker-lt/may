# Per-cluster headroom in model space: does a scene cluster predict which model wins?
# (a) per-cluster AP of M and the L-M gain per cluster (in-sample spread);
# (b) out-of-fold: pick the best model per cluster on fold A, apply to fold B; compare with a random route at the same share.
import numpy as np
from common import *
M=load('dumpml_yolo26m_coco'); L=load('dump_yolo26l_coco'); M11=load('dump_yolo11m_coco'); M12=load('dump_yolov12m_sdpa_coco')
F=feats(M); n=len(M['imgIds'])
print('M %.4f L %.4f 11m %.4f 12m %.4f'%tuple(ap_rec(x) for x in [M,L,M11,M12]))
rng=np.random.default_rng(7)
for fk in ['m640','n320']:
    c,lab=kmeans(F[fk],8,rng)
    print(f'-- {fk} K=8 in-sample per-cluster AP (size, M, L-M, 11m-M, 12m-M)')
    for k in range(8):
        s=lab==k
        a=[ap_rec(x,s) for x in [M,L,M11,M12]]
        print('  k=%d n=%4d M %.4f  L-M %+.4f  11m-M %+.4f  12m-M %+.4f'%(k,s.sum(),a[0],a[1]-a[0],a[2]-a[0],a[3]-a[0]))
def oof(recs,K,fk,seeds=range(3)):
    out=[];nul=[];shares=[]
    for sd in seeds:
        rng=np.random.default_rng(200+sd); perm=rng.permutation(n); folds=[perm[:n//2],perm[n//2:]]
        A=np.zeros(n,int)
        for f in range(2):
            tr,te=folds[f],folds[1-f]
            c,lab=kmeans(F[fk][tr],K,rng); labte=assign(F[fk][te],c)
            trm=np.zeros(n,bool)
            for k in range(K):
                m=np.zeros(n,bool); m[tr[lab==k]]=True
                best=int(np.argmax([ap_rec(x,m) for x in recs]))
                A[te[labte==k]]=best
        out.append(ap_mix(recs,A)); sh=(A>0).mean(); shares.append(sh)
        # share-matched null: random route with the same per-branch shares
        Ar=rng.permutation(A); nul.append(ap_mix(recs,Ar))
    return np.mean(out),np.mean(nul),np.mean(shares)
for name,recs in [('M/L',[M,L]),('M/11m/12m',[M,M11,M12]),('M/12m',[M,M12])]:
    for fk in ['m640','n320']:
        for K in [4,8,16,32]:
            a,nu,sh=oof(recs,K,fk)
            print(f'{name:10s} {fk} K={K:2d}: cluster-routed {a:.4f}  share-null {nu:.4f}  diff {a-nu:+.4f}  non-M share {sh:.2f}',flush=True)
