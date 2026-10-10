import numpy as np, json, contextlib, io
from scipy.stats import spearmanr
from pycocotools.coco import COCO
W='/data/tmp/ds-yolo/seminar5/work/agent4/'; D='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
with contextlib.redirect_stdout(io.StringIO()): gt=COCO(D+'instances_val2017.json')
P={k:np.load(W+f'pi_{k}.npz') for k in ['y11m','y12m','y26m','y26l']}
ids=P['y11m']['imgIds']
z=np.load(D+'val2017_stem_pooled.npz'); zi={int(i):j for j,i in enumerate(z['image_id'])}
order=np.array([zi[int(i)] for i in ids]); n320=z['n320'][order]; m640=z['m640'][order]
def iou(a,b):
    ax2,ay2=a[:,0]+a[:,2],a[:,1]+a[:,3]; bx2,by2=b[:,0]+b[:,2],b[:,1]+b[:,3]
    iw=np.clip(np.minimum(ax2[:,None],bx2[None])-np.maximum(a[:,0][:,None],b[:,0][None]),0,None)
    ih=np.clip(np.minimum(ay2[:,None],by2[None])-np.maximum(a[:,1][:,None],b[:,1][None]),0,None)
    inter=iw*ih; return inter/(a[:,2:].prod(1)[:,None]+b[:,2:].prod(1)[None]-inter+1e-9)
cnt=[];sm=[];crowd=[];ncat=[];large=[];occl=[]
for i in ids:
    an=[a for a in gt.loadAnns(gt.getAnnIds(imgIds=int(i))) if not a['iscrowd']]
    cnt.append(len(an)); sm.append(sum(a['area']<32**2 for a in an)); large.append(sum(a['area']>96**2 for a in an))
    ncat.append(len({a['category_id'] for a in an}))
    if len(an)>1:
        b=np.array([a['bbox'] for a in an],float); c=np.array([a['category_id'] for a in an])
        M=iou(b,b); np.fill_diagonal(M,0); same=(c[:,None]==c[None])
        crowd.append(((M*same).max(1)>0.3).sum()); occl.append((M.max(1)>0.3).mean())
    else: crowd.append(0); occl.append(0.)
F=dict(count=np.array(cnt),small=np.array(sm),large=np.array(large),ncat=np.array(ncat),crowd_sameclass_iou03=np.array(crowd),occl_frac=np.array(occl))
np.savez(W+'feat.npz',ids=ids,**F)
def ridge_oof(X,y,lam=10.,k=5,seed=0):
    rng=np.random.default_rng(seed); f=rng.integers(0,k,len(y)); p=np.zeros(len(y))
    X=(X-X.mean(0))/(X.std(0)+1e-6); X=np.c_[X,np.ones(len(X))]
    for j in range(k):
        tr=f!=j; A=X[tr].T@X[tr]+lam*np.eye(X.shape[1]); p[f==j]=X[f==j]@np.linalg.solve(A,X[tr].T@y[tr])
    return p
pairs={'attn(12m-11m)':('y12m','y11m'),'Ltail(26l-26m)':('y26l','y26m'),'e2e(26m-11m)':('y26m','y11m'),'attn26(12m-26m)':('y12m','y26m')}
ok=np.all([~np.isnan(P[k]['ap']) for k in P],0); print('images with GT',ok.sum())
for name,(a,b) in pairs.items():
    d=(P[a]['ap']-P[b]['ap'])[ok]
    print(f'\n== {name}: mean {d.mean():+.4f}, sd {d.std():.3f}, frac>0 {(d>0).mean():.3f} frac<0 {(d<0).mean():.3f}')
    for fn,fv in F.items():
        r=spearmanr(fv[ok],d).correlation; print(f'  spearman d vs {fn:22s} {r:+.3f}')
    # binned means by count
    c=F['count'][ok]; bins=[1,2,4,7,11,16,1000]
    print('  mean d by GT count bin:',' '.join(f'[{bins[j]},{bins[j+1]}):{d[(c>=bins[j])&(c<bins[j+1])].mean():+.4f}(n={((c>=bins[j])&(c<bins[j+1])).sum()})' for j in range(len(bins)-1)))
    for xn,X in [('n320',n320),('m640',m640)]:
        p=ridge_oof(X[ok],d,lam=100.); print(f'  OOF ridge {xn}->d spearman {spearmanr(p,d).correlation:+.3f}')
    # baseline-score confound: d vs base model's own per-image AP (not router-visible)
    print(f'  spearman d vs {b} own per-image AP (not visible): {spearmanr(P[b]["ap"][ok],d).correlation:+.3f}')
# predicted count from thumbnail / m640
lc=np.log1p(F['count'])
for xn,X in [('n320',n320),('m640',m640)]:
    p=ridge_oof(X,lc,lam=100.); print(f'\ncount predictability {xn}: spearman {spearmanr(p,lc).correlation:+.3f}')
    np.save(W+f'pred_logcount_{xn}.npy',p)
    lcr=ridge_oof(X,np.log1p(F['crowd_sameclass_iou03']),lam=100.); print(f'crowd predictability {xn}: spearman {spearmanr(lcr,F["crowd_sameclass_iou03"]).correlation:+.3f}')
    np.save(W+f'pred_crowd_{xn}.npy',lcr)
