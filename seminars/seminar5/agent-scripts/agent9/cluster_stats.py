# How separable are COCO's "domains"? Mutual information between a scene cluster and the classes of its instances,
# share of instances that are 'person', and the class overlap between clusters. Full val2017, GT-free clusters.
import numpy as np
from common import *
r=load('dumpml_yolo26m_coco'); Y=gt_matrix(r); F=feats(r); n=len(Y)
pc=Y.sum(0)/Y.sum(); Hc=-(pc*np.log2(pc+1e-12)).sum()
print('instances %d, H(class of instance) %.3f bits, person share %.3f'%(Y.sum(),Hc,pc[0]))
rng=np.random.default_rng(11)
for src in ['m640','n320','gt']:
    for K in [4,8,16,64]:
        X=F[src] if src!='gt' else (Y>0).astype(float)
        c,lab=kmeans(X,K,rng)
        J=np.array([Y[lab==k].sum(0) for k in range(K)],float); J/=J.sum()
        pk=J.sum(1,keepdims=True); pcc=J.sum(0,keepdims=True)
        MI=(J*np.log2((J+1e-12)/(pk@pcc+1e-12))).sum()
        sizes=np.bincount(lab,minlength=K)/n
        # share of each class's instances in its single most frequent cluster, instance-weighted
        dom=(J.max(0)/J.sum(0).clip(1e-12)*pcc[0]).sum()
        pers=J[:,0]/J.sum(1)
        print(f'{src:5s} K={K:2d}: I(cluster;class)={MI:.3f} bits ({MI/Hc:.1%} of H), instance share in its class-top cluster {dom:.2f}, '
              f'cluster sizes {sizes.min():.2f}-{sizes.max():.2f}, person share per cluster {pers.min():.2f}-{pers.max():.2f}')
# random-cluster MI floor
for K in [4,16,64]:
    lab=rng.integers(K,size=n); J=np.array([Y[lab==k].sum(0) for k in range(K)],float); J/=J.sum()
    pk=J.sum(1,keepdims=True); pcc=J.sum(0,keepdims=True); print('random K=%d MI %.3f bits'%(K,(J*np.log2((J+1e-12)/(pk@pcc+1e-12))).sum()))
