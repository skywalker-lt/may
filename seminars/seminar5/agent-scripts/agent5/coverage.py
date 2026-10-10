import numpy as np, sys
from tiles_common import *
gt=gt_array(); gt=gt[gt[:,7]==0]
M=load('dump_yolo26m_coco.json')
rng=np.random.default_rng(0)
small=gt[:,5]<32**2
print('images',N,'gt',len(gt),'small frac',small.mean().round(3))
for G in (5,10,16):
    cm=content_mask(G); tg=tile_of(gt,G); tm=tile_of(M,G)
    cnt=np.zeros((N,G*G)); np.add.at(cnt,(gt[:,0].astype(int),tg),1)
    cnts=np.zeros((N,G*G)); np.add.at(cnts,(gt[small,0].astype(int),tg[small]),1)
    occ=(cnt>0).sum(1)/(G*G)
    # router proxies from M's own detections
    for thr in (0.1,0.3):
        pass
    sm=M[:,5]>=0.1
    rm=np.zeros((N,G*G)); np.add.at(rm,(M[sm,0].astype(int),tm[sm]),M[sm,5])
    rand=rng.random((N,G*G)); randc=rand+cm*1.0  # random among content tiles first
    print(f'G={G}: mean share of tiles holding >=1 GT centre {occ.mean():.3f} (median {np.median(occ):.3f}); content tiles {cm.mean():.3f}')
    for s in (0.10,0.16,0.25,0.40):
        k=max(1,int(round(s*G*G)))
        out=[]
        for nm,score in (('GTcount',cnt+1e-6*rand),('Mdet',rm+1e-6*rand),('random',rand),('random-content',randc)):
            sel=np.argsort(-score,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1)
            cov=S[gt[:,0].astype(int),tg].mean(); covs=S[gt[small,0].astype(int),tg[small]].mean()
            out.append(f'{nm} {cov:.3f}/{covs:.3f}')
        print(f'  share {s:.2f} (k={k}) objects covered all/small: '+' | '.join(out))
