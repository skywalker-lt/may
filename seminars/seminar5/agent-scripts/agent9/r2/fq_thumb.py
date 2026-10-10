# Add the GT-free thumbnail count router (n320 / m640 ridge to log(1+GT count), fitted on the 4,500 val images outside the subset)
import sys, json, numpy as np
exec(open('fq_analyse.py').read().split("pf=perimg(recs['fp'])")[0].replace("print('subset","(lambda *a,**k:None)('subset"))
z=np.load('/data/tmp/ds-yolo/seminar5/inputs/dumps/val2017_stem_pooled.npz'); ids=list(z['image_id']); pos={im:i for i,im in enumerate(ids)}
with contextlib.redirect_stdout(io.StringIO()):
    from pycocotools.coco import COCO as _C
cntall={im:0 for im in ids}
for a in gt.dataset['annotations']:
    if not a.get('iscrowd',0): cntall[a['image_id']]+=1
yall=np.log1p(np.array([cntall[im] for im in ids])); insub=np.isin(np.array(ids),np.array(sub))
for fk in ('n320','m640'):
    X=z[fk].astype(np.float64); X=(X-X.mean(0))/(X.std(0)+1e-6); X=np.c_[X,np.ones(len(X))]
    tr=~insub; w=np.linalg.solve(X[tr].T@X[tr]+30*np.eye(X.shape[1]),X[tr].T@yall[tr])
    pred=X[[pos[im] for im in sub]]@w
    rng=np.random.default_rng(1)
    for q in ('max','p9999'):
        d=ap_rec(recs['fp'])-ap_rec(recs[q])
        for s in (0.2,0.3):
            k=int(s*n); A=np.zeros(n,int); A[np.argsort(-pred)[:k]]=1; a=ap_mix([recs[q],recs['fp']],A)
            nul=np.mean([ap_mix([recs[q],recs['fp']],rng.permutation(A)) for _ in range(20)])
            print(f'{fk} predicted-count router [{q}] fp share {s}: {a:.4f} null {nul:.4f} g {a-nul:+.4f} r {(a-nul)/((1-s)*d):.2f}',flush=True)
