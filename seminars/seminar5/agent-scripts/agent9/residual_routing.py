# Label-space mixture of domains, fitted as a RESIDUAL on the detector's own score:
#   score' = sigmoid(a_c*logit(s) + b_c + d[route(img), c])
# a_c, b_c: per-class Platt calibration (monotone per class: leaves AP unchanged by itself);
# d: per-(route, class) offset fitted on fold A (soft TP target = share of the 10 IoU thresholds matched), ridge lam;
# applied to fold B. 2-fold, 3 random splits. Routes: k-means on GT-free image features, random (null), kNN (dense),
# GT-presence clusters (oracle route, a bound).
import sys, numpy as np
from common import *
r=load('dumpml_yolo26m_coco'); Y=gt_matrix(r); P=(Y>0).astype(float); F=feats(r); n=len(P)
base=ap_rec(r); print('base M AP %.4f'%base,flush=True)
s=np.clip(r['score'],1e-6,1-1e-6); u=np.log(s/(1-s)); cat=r['cat'].astype(int); img=r['img']
valid=~r['ign'][0]; y=(r['match']&~r['ign']).mean(0)
# GT-free image descriptor from M's own output: max score per class (the head's class histogram)
H=np.zeros((n,80)); np.maximum.at(H,(img,cat),s); F['dethist']=(H-H.mean(0))/(H.std(0)+1e-6)
sig=lambda x:1/(1+np.exp(-x))
def platt(sel):
    a=np.ones(80); b=np.zeros(80)
    for it in range(8):
        z=a[cat]*u+b[cat]; p=sig(z); g=(y-p)*valid*sel; h=p*(1-p)*valid*sel+1e-9
        # Newton per class on (a,b), 2x2
        Gu=np.bincount(cat,g*u,80); Gb=np.bincount(cat,g,80)
        Huu=np.bincount(cat,h*u*u,80)+1e-3; Hub=np.bincount(cat,h*u,80); Hbb=np.bincount(cat,h,80)+1e-3
        det=Huu*Hbb-Hub**2; a+=(Hbb*Gu-Hub*Gb)/det; b+=(Huu*Gb-Hub*Gu)/det
    return a,b
def offsets(route_det,K,sel,a,b,lam):
    d=np.zeros((K,80)); key=route_det*80+cat
    for it in range(4):
        p=sig(a[cat]*u+b[cat]+d.ravel()[key]); w=valid*sel
        g=np.bincount(key,(y-p)*w,K*80); h=np.bincount(key,p*(1-p)*w,K*80)
        d+=(g/(h+lam)).reshape(K,80)
    return d
A0,B0=None,None
def evaluate(route_fn,lams=(30,300),seeds=(0,1,2)):
    res={l:[] for l in lams}
    for sd in seeds:
        rng=np.random.default_rng(300+sd); perm=rng.permutation(n); folds=[perm[:n//2],perm[n//2:]]
        Z={l:np.zeros(len(s)) for l in lams}
        for f in range(2):
            tr,te=folds[f],folds[1-f]; trm=np.zeros(n,bool); trm[tr]=True; tem=~trm
            sel=trm[img].astype(float)
            a,b=A0,B0  # one global per-class Platt map (AP-neutral by itself); only d is fold-fitted
            route,K=route_fn(tr,te,rng)   # route: (n,) int for all images (train ones from train fit)
            for l in lams:
                d=offsets(route[img],K,sel,a,b,l)
                zt=a[cat]*u+b[cat]+d.ravel()[route[img]*80+cat]
                Z[l][tem[img]]=zt[tem[img]]
        for l in lams: res[l].append(ap_rec(r,score=sig(Z[l])))
    return {l:(np.mean(v),np.std(v)) for l,v in res.items()}
def show(name,res):
    print(f'{name:38s} '+'  '.join('lam=%d: %.4f (%+.4f, sd %.4f)'%(l,m,m-base,sd) for l,(m,sd) in res.items()),flush=True)
def km_route(X,K):
    def fn(tr,te,rng):
        c,lab=kmeans(X[tr],K,rng); route=np.zeros(n,int); route[tr]=lab; route[te]=assign(X[te],c); return route,K
    return fn
def rand_route(K):
    def fn(tr,te,rng): return rng.integers(K,size=n),K
    return fn
A0,B0=platt(np.ones(len(s)))
print('Platt-only AP %.4f'%ap_rec(r,score=sig(A0[cat]*u+B0[cat])),flush=True)
show('K=1 (per-class Platt only)',evaluate(lambda tr,te,rng:(np.zeros(n,int),1)))
for K in [4,16,64]:
    show(f'null random K={K}',evaluate(rand_route(K)))
    for fk in ['m640','n320','dethist']:
        show(f'scene K={K} router={fk}',evaluate(km_route(F[fk],K)))
    show(f'GT-presence clusters K={K} (oracle route)',evaluate(km_route(P,K)))
