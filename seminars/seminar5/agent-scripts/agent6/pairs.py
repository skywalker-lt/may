import sys, json, numpy as np, time
from mixacc import Mix
from scipy.stats import spearmanr
A,B,tag=sys.argv[1],sys.argv[2],sys.argv[3]
t=time.time(); M=Mix(A,B); print('load',round(time.time()-t,1),flush=True)
nI=len(M.imgIds)
print('A',round(M.ap(np.zeros(nI)),4),'B',round(M.ap(np.ones(nI)),4),flush=True)
t=time.time(); g=M.marginal(0); print('marginal time',round(time.time()-t,1),flush=True)
np.save(f'gain_{tag}.npy',g)
INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
z=np.load(INP+'val2017_stem_pooled.npz'); order=np.array([M.img_index[i] for i in z['image_id']])
N320=np.zeros((nI,128)); N320[order]=z['n320']; M640=np.zeros((nI,512)); M640[order]=z['m640']
gt=json.load(open(INP+'instances_val2017.json')); cnt=np.zeros(nI); small=np.zeros(nI)
for a in gt['annotations']:
    if a.get('iscrowd',0): continue
    i=M.img_index[a['image_id']]; cnt[i]+=1; small[i]+= a['area']<32**2
rng=np.random.RandomState(0); folds=rng.permutation(nI)%5
def ridge_oof(X,y,lam=10.0):
    p=np.zeros(nI)
    for f in range(5):
        tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd
        A_=Xs[tr]; w=np.linalg.solve(A_.T@A_+lam*len(A_)*np.eye(X.shape[1])/100, A_.T@(y[tr]-y[tr].mean()))
        p[~tr]=Xs[~tr]@w+y[tr].mean()
    return p
for share in (0.1,0.25,0.5):
    n=int(share*nI)
    orc=np.zeros(nI,np.int8); orc[np.argsort(-g)[:n]]=1
    nulls=[]
    for d in range(3):
        s=np.zeros(nI,np.int8); s[rng.permutation(nI)[:n]]=1; nulls.append(M.ap(s))
    res={'oracle':M.ap(orc),'null':np.mean(nulls)}
    for nm,f in (('GTcount',cnt),('GTsmall',small),('-GTcount',-cnt)):
        s=np.zeros(nI,np.int8); s[np.argsort(-f+1e-9*rng.rand(nI))[:n]]=1; res[nm]=M.ap(s)
    for nm,X in (('n320',N320),('m640',M640)):
        p=ridge_oof(X,g); s=np.zeros(nI,np.int8); s[np.argsort(-p)[:n]]=1; res['learned_'+nm]=M.ap(s)
        if share==0.1: print(' rho(pred gain, gain)',nm, round(spearmanr(p,g).correlation,3))
    print(f'{tag} B-share {share}: '+' '.join(f'{k}={v:.4f}' for k,v in res.items()),flush=True)
print(f'{tag} oracle unconstrained (gain>0 share {np.mean(g>0):.2f})', round(M.ap((g>0).astype(np.int8)),4))
print(' rho(gain, GT count)', round(spearmanr(g,cnt).correlation,3), ' rho(gain, GT small count)', round(spearmanr(g,small).correlation,3))
