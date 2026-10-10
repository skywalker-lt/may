import json, pickle, numpy as np
from fastap import ap_rec, ap_mix, ap_from
D='/data/tmp/ds-yolo/seminar5/inputs/dumps'
C='/data/tmp/ds-yolo/seminar5/work/agent9/cache'
def load(n): return pickle.load(open(f'{C}/{n}.pkl','rb'))
def gt_matrix(rec):
    g=json.load(open(f'{D}/instances_val2017.json'))
    ii={im:i for i,im in enumerate(rec['imgIds'])}; kk={c:k for k,c in enumerate(rec['catIds'])}
    Y=np.zeros((len(ii),80),np.int32); A=np.zeros((len(ii),80))
    for a in g['annotations']:
        if a.get('iscrowd',0): continue
        Y[ii[a['image_id']],kk[a['category_id']]]+=1
    return Y
def feats(rec):
    z=np.load(f'{D}/val2017_stem_pooled.npz'); ids=list(z['image_id']); pos={im:i for i,im in enumerate(ids)}
    o=np.array([pos[im] for im in rec['imgIds']])
    out={}
    for k in ['n320','m640']:
        X=z[k][o].astype(np.float64); X=(X-X.mean(0))/(X.std(0)+1e-6); out[k]=X
    return out
def kmeans(X,K,rng,iters=50,restarts=3):
    best=None
    for r in range(restarts):
        c=[X[rng.integers(len(X))]]; dmin=((X-c[0])**2).sum(1)
        for _ in range(K-1):
            c.append(X[rng.choice(len(X),p=dmin/dmin.sum())]); dmin=np.minimum(dmin,((X-c[-1])**2).sum(1))
        c=np.array(c)
        for it in range(iters):
            d=((X**2).sum(1)[:,None]-2*X@c.T+(c**2).sum(1)[None]); lab=d.argmin(1)
            cnt=np.bincount(lab,minlength=K); sm=np.zeros_like(c); np.add.at(sm,lab,X); nc=np.where(cnt[:,None]>0,sm/np.maximum(cnt,1)[:,None],c)
            if np.allclose(nc,c): break
            c=nc
        inert=d[np.arange(len(X)),lab].sum()
        if best is None or inert<best[0]: best=(inert,c,lab)
    return best[1],best[2]
def assign(X,c): return ((X**2).sum(1)[:,None]-2*X@c.T+(c**2).sum(1)[None]).argmin(1)
