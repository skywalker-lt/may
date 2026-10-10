import json, numpy as np, glob
from fastacc import FastAcc
np.random.seed(1)
INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
F=FastAcc('evm.pkl'); nI=len(F.imgIds); nK=len(F.catIds)
gt=json.load(open(INP+'instances_val2017.json'))
Y=np.zeros((nI,nK))
for a in gt['annotations']:
    if not a.get('iscrowd',0): Y[F.img_index[a['image_id']],F.cat_index[a['category_id']]]=1
lg=lambda p: np.log(np.clip(p,1e-6,1-1e-6))-np.log1p(-np.clip(p,1e-6,1-1e-6))
def auc(z,y):
    # mean per-class AUC
    out=[]
    for k in range(y.shape[1]):
        pos=z[y[:,k]>0,k]; neg=z[y[:,k]==0,k]
        if len(pos)==0: continue
        r=np.argsort(np.argsort(np.concatenate([pos,neg])))+1
        out.append((r[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg)))
    return np.mean(out)
for f in sorted(glob.glob('oof_*.npy')):
    Z=np.load(f); P=1/(1+np.exp(-Z))
    res=[]
    for b in (0.05,0.1,0.2):
        res.append(('s*p^%g'%b, F.ap(factor=P**b)))
    for b in (0.05,0.1,0.2):
        res.append(('logit+%g(z-prior)'%b, F.ap(logit_add=b*(Z-lg(Y.mean(0))))))
    best=max(res,key=lambda t:t[1])
    print(f, 'AUC %.3f'%auc(Z,Y), ' '.join(f'{n}:{a:.4f}' for n,a in res))
S=None
