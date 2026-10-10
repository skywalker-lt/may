import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, sys
sys.argv=['x']; 
from qroute import load, match, N
for bn,en in [('M640','L640'),('L544','L640'),('L544','X640')]:
    B=load(bn); E=load(en); m=match(B,E,bn,en); ok=m>=0; s=B[:,5]; Bi=B[:,0].astype(int)
    Rn=np.where(ok,E[np.maximum(m,0),5],0.1*s); d=np.abs(Rn-s)
    bb=np.searchsorted(Bi,np.arange(N+1)); cnt=np.diff(bb)
    # per-image: fraction of |dScore| mass in the top-k by s(1-s) and by random
    unc=s*(1-s); rng=np.random.default_rng(0); rnd=rng.random(len(B))
    out={}
    for k in (32,64,128):
        cu=cr=0.0
        for i in range(N):
            A=np.arange(bb[i],bb[i+1]); 
            if len(A)==0: continue
            cu+=d[A[np.argsort(-unc[A])[:k]]].sum(); cr+=d[A[np.argsort(-rnd[A])[:k]]].sum()
        out[k]=(cu/d.sum(),cr/d.sum())
    # match rate by score band; mean |dScore| by band
    bands=[(0,0.05),(0.05,0.15),(0.15,0.3),(0.3,0.5),(0.5,0.7),(0.7,1.01)]
    print(f'== base {bn} expert {en}: rows {len(B)}, per-image candidates mean {cnt.mean():.1f}; matched {ok.mean():.3f}; cands with s>=0.05 per image {(s>=0.05).sum()/N:.1f}, s>=0.1 {(s>=0.1).sum()/N:.1f}, s>=0.25 {(s>=0.25).sum()/N:.1f}')
    for lo,hi in bands:
        w=(s>=lo)&(s<hi); print(f'   band [{lo},{hi}): n/img {w.sum()/N:6.1f} matched {ok[w].mean():.3f} mean|dScore| {d[w].mean():.3f} share of dScore mass {d[w].sum()/d.sum():.3f}')
    for k,(a,b) in out.items(): print(f'   k={k}: dScore mass captured unc {a:.3f} random {b:.3f}')
    # matched pairs: IoU between B box and E box (geometry change)
    bbx=B[ok,1:5]; ee=E[m[ok],1:5]; ix=np.clip(np.minimum(bbx[:,0]+bbx[:,2],ee[:,0]+ee[:,2])-np.maximum(bbx[:,0],ee[:,0]),0,None); iy=np.clip(np.minimum(bbx[:,1]+bbx[:,3],ee[:,1]+ee[:,3])-np.maximum(bbx[:,1],ee[:,1]),0,None)
    inter=ix*iy; io=inter/(bbx[:,2]*bbx[:,3]+ee[:,2]*ee[:,3]-inter)
    print(f'   matched pairs IoU: mean {io.mean():.3f}, <0.75 share {(io<0.75).mean():.3f}, <0.9 share {(io<0.9).mean():.3f}')
