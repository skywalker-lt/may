"""Object-consistent tile assignment for the expensive model's detections: an E detection takes the tile of its best
same-class M match (IoU>=0.5) if one exists, else its own centre tile. Cached per (E, G)."""
import numpy as np, os
from tiles_common import *
def iou(a,b):
    ax2=a[:,None,0]+a[:,None,2]; ay2=a[:,None,1]+a[:,None,3]; bx2=b[None,:,0]+b[None,:,2]; by2=b[None,:,1]+b[None,:,3]
    iw=np.clip(np.minimum(ax2,bx2)-np.maximum(a[:,None,0],b[None,:,0]),0,None); ih=np.clip(np.minimum(ay2,by2)-np.maximum(a[:,None,1],b[None,:,1]),0,None)
    inter=iw*ih; return inter/(a[:,None,2]*a[:,None,3]+b[None,:,2]*b[None,:,3]-inter+1e-9)
def anchored_tiles(M,E,G,tag):
    f=f'/data/tmp/ds-yolo/seminar5/work/agent5/cache_anch_{tag}_{G}.npy'
    if os.path.exists(f): return np.load(f)
    tm=tile_of(M,G); te=tile_of(E,G).copy()
    Mi=M[:,0].astype(int); Ei=E[:,0].astype(int)
    mo=np.argsort(Mi,kind='stable'); eo=np.argsort(Ei,kind='stable')
    mb=np.searchsorted(Mi[mo],np.arange(N+1)); eb=np.searchsorted(Ei[eo],np.arange(N+1))
    for i in range(N):
        a=mo[mb[i]:mb[i+1]]; b=eo[eb[i]:eb[i+1]]
        if len(a)==0 or len(b)==0: continue
        U=iou(E[b,1:5],M[a,1:5])*(E[b,6][:,None]==M[a,6][None,:])
        j=U.argmax(1); ok=U.max(1)>=0.5
        te[b[ok]]=tm[a[j[ok]]]
    np.save(f,te); return te
