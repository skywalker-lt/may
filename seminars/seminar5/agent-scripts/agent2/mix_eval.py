# Exact mixture APs for depth routing between YOLO26-M (shallow) and YOLO26-L (deep), public weights, val2017.
import os, json, pickle, time, numpy as np, copy, io, contextlib
os.environ['OMP_NUM_THREADS']='1'
from scipy.stats import spearmanr
from mixlib import *
from pycocotools.cocoeval import COCOeval
W='/data/tmp/ds-yolo/seminar5/work/agent2'
M=load('dumpml_yolo26m_coco'); L=load('dump_yolo26l_coco'); ids=np.array(M['imgIds']); I=len(ids)
sig=pickle.load(open(f'{W}/cache/signal.pkl','rb')); gain=sig['gain']; F=np.load(f'{W}/cache/featM.npy'); names=sig['names']
g=gt(); ngt=np.array([sum(1 for a in g.imgToAnns[int(i)] if not a['iscrowd']) for i in ids]); nsmall=np.array([sum(1 for a in g.imgToAnns[int(i)] if not a['iscrowd'] and a['area']<32**2) for i in ids])
z=np.load(f'{D}/val2017_stem_pooled.npz')
def oof(X,y,lam=10.,k=5,seed=0):
    rng=np.random.RandomState(seed); fold=rng.randint(0,k,len(y)); p=np.zeros(len(y))
    for f in range(k):
        tr=fold!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xt=(X[tr]-mu)/sd; Xv=(X[~tr]-mu)/sd
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(X.shape[1]),Xt.T@(y[tr]-y[tr].mean())); p[~tr]=Xv@w
    return p
y=np.log1p(ngt)
R={'oracle (per-image gain)':gain,'GT count rule':ngt+1e-3*np.random.RandomState(1).rand(I),
   'M-output n>0.25 (late, GT-free)':F[:,names.index('n>0.25')]+1e-3*np.random.RandomState(2).rand(I),
   'M-output ridge->count (late)':oof(F,y,10.),'stem m640 ridge->count':oof(z['m640'],y,100.),'thumb n320 ridge->count':oof(z['n320'],y,100.),
   'M-output ridge->gain (late)':oof(F,gain,10.)}
for k,v in R.items():
    if 'oracle' not in k: print(f'  {k:34s} Spearman with GT count {spearmanr(v,ngt).correlation:+.3f}')
res={}
for s in (0.16,0.31,0.42):
    n=int(round(s*I)); print(f'--- L share {s} ({n} images)',flush=True)
    for k,v in R.items():
        ch=np.zeros(I,int); ch[np.argsort(-v)[:n]]=1; ap,sml=mix_ap([M,L],ch); res[(s,k)]=ap; print(f'  {k:34s} AP {ap:.4f}  S/M/L {sml[0]:.4f}/{sml[1]:.4f}/{sml[2]:.4f}',flush=True)
    nl=[]
    for d in range(3):
        ch=np.zeros(I,int); ch[np.random.RandomState(100+d).permutation(I)[:n]]=1; nl.append(mix_ap([M,L],ch)[0])
    res[(s,'null')]=np.mean(nl); print(f'  share null (3 draws)                AP {np.mean(nl):.4f}  draws {" ".join("%.4f"%x for x in nl)}',flush=True)
pickle.dump(res,open(f'{W}/cache/mix_res.pkl','wb'))
# AP of M and L on subsets by GT count
def ap_subset(cache,mask):
    I=len(cache['imgIds']); K=len(cache['catIds']); A=4; sub=np.where(mask)[0]
    ev=[cache['evalImgs'][k*A*I+a*I+i] for k in range(K) for a in range(A) for i in sub]
    E=COCOeval(g,None,'bbox'); E.params.imgIds=[cache['imgIds'][i] for i in sub]; E.params.catIds=cache['catIds']; E.evalImgs=ev; E._paramsEval=copy.deepcopy(E.params)
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0]
print('--- AP by GT-count bucket (images, instances): M, L, L-M')
for lo,hi in ((0,1),(1,3),(3,6),(6,11),(11,10**6)):
    m=(ngt>=lo)&(ngt<hi)
    if m.sum()==0: continue
    a,b=ap_subset(M,m),ap_subset(L,m); print(f'  n_gt in [{lo},{hi}): {m.sum():4d} img, {ngt[m].sum():5d} inst ({ngt[m].sum()/ngt.sum():.2f})  M {a:.4f} L {b:.4f} gain {b-a:+.4f}',flush=True)
