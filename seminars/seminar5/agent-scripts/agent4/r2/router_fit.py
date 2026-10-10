"""Fit the shrink router on 12,000 train2017 images (engine-identical features), freeze it, score val2017 (direction 1's K2 gate).
Target log1p(#boxes with sqrt(w*h) < 96 px, original pixels) ~ COCO small+medium. Threshold = train quantile for share 0.5 / 0.6."""
import os, sys; os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent1')
import numpy as np, mixlib as M, feats as F
from scipy.stats import spearmanr
W='/data/tmp/ds-yolo/seminar5/work/agent4/r2/'
tr=np.load(W+'rfeat_train12k.npz'); va=np.load(W+'rfeat_val.npz')
X=tr['F'].astype(np.float64); y=np.log1p(tr['cnt'][:,1]+tr['cnt'][:,2])
mu,sd=X.mean(0),X.std(0)+1e-6; Xs=(X-mu)/sd; lam=10.0
wz=np.linalg.solve(Xs.T@Xs+lam*len(Xs)/100*np.eye(128),Xs.T@(y-y.mean()))
w=wz/sd; b=y.mean()-mu@w
s_tr=X@w+b; ids,ns,nm,nl,_=F.gt_counts(); vi={int(i):k for k,i in enumerate(va['ids'])}; Xv=va['F'][[vi[int(i)] for i in ids]].astype(np.float64); s_va=Xv@w+b
print('train spearman', round(spearmanr(s_tr,y).correlation,3), 'val spearman vs GT S+M', round(spearmanr(s_va,ns+nm).correlation,3))
m512,m640=M.evaluated('dumpml_yolo26m_512_coco'),M.evaluated('dumpml_yolo26m_coco')
for q in (0.4,0.5,0.6):
    thr=np.quantile(s_tr,q); sh=s_va<thr
    print(f'share {q}: train thr {thr:.3f} -> realised val share {sh.mean():.3f}; RS-2 AP with train-fitted router {M.score([m512,m640],np.where(sh,0,1))}',flush=True)
    if q==0.5: np.savez(W+'router_train.npz',w=w,b=b,thr=thr,s_va=s_va,ids=ids)
