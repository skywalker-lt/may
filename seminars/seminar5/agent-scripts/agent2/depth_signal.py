# Does anything ground-truth-free predict which image gains from depth (L over M)?  Features: (a) M's own detections
# (the richest signal a late / self-assessing router could see), (b) pooled stem features (m640 layer 5, n320 thumbnail).
import os, json, pickle, numpy as np, collections
os.environ['OMP_NUM_THREADS']='1'
from scipy.stats import spearmanr
from mixlib import *
W='/data/tmp/ds-yolo/seminar5/work/agent2'
M=load('dumpml_yolo26m_coco'); L=load('dump_yolo26l_coco'); ids=np.array(M['imgIds'])
assert ids.tolist()==L['imgIds']
if os.path.exists(f'{W}/cache/perimg.npz'):
    z=np.load(f'{W}/cache/perimg.npz'); apM,apL=z['apM'],z['apL']
else:
    apM=per_image_ap(M); apL=per_image_ap(L); np.savez(f'{W}/cache/perimg.npz',apM=apM,apL=apL)
gain=np.nan_to_num(apL-apM)
print('per-image AP mean M %.4f L %.4f; gain>0 %.3f gain<0 %.3f zero %.3f'%(np.nanmean(apM),np.nanmean(apL),(gain>0).mean(),(gain<0).mean(),(gain==0).mean()))
# --- features from M's own output
g=gt(); dets=collections.defaultdict(list)
for d in json.load(open(f'{D}/dumpml_yolo26m_coco.json')): dets[d['image_id']].append((d['score'],*d['bbox'],d['category_id']))
F=[];names=None
def iou(a,b):
    x1=np.maximum(a[:,None,0],b[None,:,0]); y1=np.maximum(a[:,None,1],b[None,:,1]); x2=np.minimum(a[:,None,0]+a[:,None,2],b[None,:,0]+b[None,:,2]); y2=np.minimum(a[:,None,1]+a[:,None,3],b[None,:,1]+b[None,:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None); return inter/(a[:,None,2]*a[:,None,3]+b[None,:,2]*b[None,:,3]-inter+1e-9)
for i in ids:
    a=np.array(dets.get(int(i),[(0,0,0,1,1,-1)]),float); s=a[:,0]; ar=a[:,3]*a[:,4]; info=g.imgs[int(i)]; A=info['width']*info['height']
    f=collections.OrderedDict()
    for t in (0.05,0.1,0.25,0.5,0.7): f[f'n>{t}']=np.log1p((s>t).sum())
    f['sum_s']=np.log1p(s.sum()); f['max_s']=s.max(); f['top5']=np.sort(s)[::-1][:5].mean()
    f['amb']=np.log1p(((s>0.1)&(s<0.4)).sum())-np.log1p((s>=0.4).sum())
    hi=s>0.25
    f['n_small']=np.log1p((hi&(ar<32**2)).sum()); f['n_med']=np.log1p((hi&(ar>=32**2)&(ar<96**2)).sum()); f['n_large']=np.log1p((hi&(ar>=96**2)).sum())
    f['frac_small']=(hi&(ar<32**2)).sum()/max(hi.sum(),1)
    f['ncls']=len(set(a[hi,5].tolist())); f['area_cov']=min(ar[hi].sum()/A,5) if hi.any() else 0
    # class confusion: overlapping boxes of different classes among score>0.1
    m=s>0.1; b=a[m][:100]
    if len(b)>1:
        o=iou(b[:,1:5],b[:,1:5]); diff=b[:,5][:,None]!=b[:,5][None,:]; f['confuse']=np.log1p(np.triu((o>0.7)&diff,1).sum()); f['crowd']=np.log1p(np.triu((o>0.3)&~diff,1).sum())
    else: f['confuse']=0; f['crowd']=0
    f['ent']=float(-(s[s>0.05]*np.log(s[s>0.05])).sum()) if (s>0.05).any() else 0
    f['imgA']=A/1e5; names=list(f); F.append(list(f.values()))
F=np.array(F,float); np.save(f'{W}/cache/featM.npy',F); json.dump(names,open(f'{W}/cache/featM_names.json','w'))
print('\nSpearman of single GT-free features with the per-image gain (L - M), all 5000 / images with GT:')
has=~np.isnan(apM)
for j,n in enumerate(names):
    r1=spearmanr(F[:,j],gain).correlation; r2=spearmanr(F[has,j],gain[has]).correlation; r3=spearmanr(F[has,j],np.abs(gain[has])).correlation
    print(f'  {n:10s} rho(gain) {r1:+.3f}  with-GT {r2:+.3f}  rho(|gain|) {r3:+.3f}')
# ridge, out-of-fold, on M-output features, on stem features, on both
z=np.load(f'{D}/val2017_stem_pooled.npz'); assert (z['image_id']==ids).all()
def oof(X,y,lam=10.,k=5,seed=0):
    rng=np.random.RandomState(seed); fold=rng.randint(0,k,len(y)); p=np.zeros(len(y))
    for f in range(k):
        tr=fold!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xt=(X[tr]-mu)/sd; Xv=(X[~tr]-mu)/sd
        w=np.linalg.solve(Xt.T@Xt+lam*len(Xt)*1e-3*np.eye(X.shape[1]),Xt.T@(y[tr]-y[tr].mean())); p[~tr]=Xv@w
    return p
sets={'M-output':F,'stem m640':z['m640'],'thumb n320':z['n320'],'M-output+m640':np.hstack([F,z['m640']])}
P={}
print('\nout-of-fold ridge predicting the gain: Spearman with the true gain')
for n,X in sets.items():
    for lam in (1.,10.,100.):
        p=oof(X,gain,lam); print(f'  {n:16s} lam {lam:5.0f} rho {spearmanr(p,gain).correlation:+.3f}'); P[(n,lam)]=p
pickle.dump({'gain':gain,'P':P,'names':names},open(f'{W}/cache/signal.pkl','wb'))
