# Headroom of an image-level routed classifier (per-image, per-class logit bias) on YOLO26-M's val2017 dump.
import json, numpy as np, sys, time
from scipy.optimize import minimize
from fastacc import FastAcc
np.random.seed(0)
INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
dumpname = sys.argv[1] if len(sys.argv)>1 else 'dumpml_yolo26m_coco.json'
pkl = sys.argv[2] if len(sys.argv)>2 else 'evm.pkl'
F = FastAcc(pkl); nI=len(F.imgIds); nK=len(F.catIds)
gt = json.load(open(INP+'instances_val2017.json'))
Y = np.zeros((nI,nK))
for a in gt['annotations']:
    if a.get('iscrowd',0): continue
    Y[F.img_index[a['image_id']], F.cat_index[a['category_id']]] = 1
dets = json.load(open(INP+dumpname))
S = np.full((nI,nK), 1e-4)
for d in dets:
    i=F.img_index[d['image_id']]; k=F.cat_index[d['category_id']]
    if d['score']>S[i,k]: S[i,k]=d['score']
z = np.load(INP+'val2017_stem_pooled.npz')
order = np.array([F.img_index[i] for i in z['image_id']]); 
M640 = np.zeros((nI,512)); M640[order]=z['m640']; N320=np.zeros((nI,128)); N320[order]=z['n320']
lg = lambda p: np.log(np.clip(p,1e-6,1-1e-6))-np.log1p(-np.clip(p,1e-6,1-1e-6))
def stdz(X, tr): mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; return (X-mu)/sd
def fit_logreg(X, Yt, lam):
    n,d=X.shape; K=Yt.shape[1]
    def f(w):
        W=w[:d*K].reshape(d,K); b=w[d*K:]
        Z=X@W+b; P=1/(1+np.exp(-Z))
        loss = np.sum(np.logaddexp(0,Z) - Yt*Z)/n + lam*np.sum(W*W)
        G=(P-Yt)/n
        return loss, np.concatenate([(X.T@G+2*lam*W).ravel(), G.sum(0)])
    w0=np.zeros(d*K+K); w0[d*K:]=lg(Yt.mean(0)*0.98+0.01)
    r=minimize(f,w0,jac=True,method='L-BFGS-B',options=dict(maxiter=300))
    return r.x[:d*K].reshape(d,K), r.x[d*K:]
folds = np.random.permutation(nI)%5
def oof(X, lam):
    Pz=np.zeros((nI,nK))
    for f in range(5):
        tr=folds!=f; te=folds==f
        Xs=stdz(X,tr); W,b=fit_logreg(Xs[tr],Y[tr],lam); Pz[te]=Xs[te]@W+b
    return Pz  # OOF logits
def report(name, Z, betas=(0.25,0.5,1.0)):
    prior = lg(Y.mean(0))
    out=[]
    for bta in betas:
        out.append((bta, F.ap(logit_add=bta*(Z-prior))))
    perm=np.random.permutation(nI)
    best=max(out,key=lambda t:t[1])
    nullv = np.mean([F.ap(logit_add=best[0]*(Z[np.random.permutation(nI)]-prior)) for _ in range(3)])
    print(f'{name:40s} ' + ' '.join(f'b{b}:{a:.4f}' for b,a in out) + f' | null(perm, b={best[0]}) {nullv:.4f}', flush=True)
    return out, nullv
base=F.ap(); print('base', round(base,4))
# oracle bounds
for big in (1.0,2.0,4.0,9.0):
    print('oracle presence logit +-%g'%big, round(F.ap(logit_add=np.where(Y>0,big,-big)),4))
print('oracle hard (absent classes x1e-3)', round(F.ap(factor=np.where(Y>0,1,1e-3)),4))
print('oracle only demote absent by -2', round(F.ap(logit_add=np.where(Y>0,0,-2.0)),4))
t=time.time()
for name,X,lam in [('n320 thumbnail (128)',N320,1e-3),('m640 stem (512)',M640,1e-3),
                   ('own max-score logits (80)',lg(S),1e-3),('own max-score + m640',np.hstack([lg(S),M640]),1e-3)]:
    Z=oof(X,lam); 
    auc=None
    report(name,Z)
    np.save(f'oof_{name.split()[0]}_{name.split()[1]}.npy',Z)
print('time',time.time()-t)
