"""How inconsistent is centre-tile mixing? Count cross-source duplicates (same class, IoU>=0.6, both score>=0.3) and
objects lost (a confident M det in an unselected tile whose L counterpart's centre lies in a selected tile, or vice versa)."""
from mix_eval import *
M=load(MODELS['M']); L=load(MODELS['L']); G=10; rng=np.random.default_rng(0)
def iou(a,b):
    ax2=a[:,None,0]+a[:,None,2]; ay2=a[:,None,1]+a[:,None,3]; bx2=b[None,:,0]+b[None,:,2]; by2=b[None,:,1]+b[None,:,3]
    iw=np.clip(np.minimum(ax2,bx2)-np.maximum(a[:,None,0],b[None,:,0]),0,None); ih=np.clip(np.minimum(ay2,by2)-np.maximum(a[:,None,1],b[None,:,1]),0,None)
    inter=iw*ih; return inter/(a[:,None,2]*a[:,None,3]+b[None,:,2]*b[None,:,3]-inter+1e-9)
for rule in ('Mdet0.01','random'):
    sc=scores(rule,G,M,L,rng); sel=np.argsort(-sc,1)[:,:16]; S=np.zeros((N,100),bool); np.put_along_axis(S,sel,True,1)
    sm=S[M[:,0].astype(int),tile_of(M,G)]; sl=S[L[:,0].astype(int),tile_of(L,G)]
    dup=lost=pairs=0
    Mi=M[:,0].astype(int); Li=L[:,0].astype(int)
    mo=np.argsort(Mi,kind='stable'); lo=np.argsort(Li,kind='stable')
    mb=np.searchsorted(Mi[mo],np.arange(N+1)); lb=np.searchsorted(Li[lo],np.arange(N+1))
    for i in range(N):
        a=mo[mb[i]:mb[i+1]]; b=lo[lb[i]:lb[i+1]]
        a=a[M[a,5]>=0.3]; b=b[L[b,5]>=0.3]
        if len(a)==0 or len(b)==0: continue
        U=iou(M[a,1:5],L[b,1:5])*(M[a,6][:,None]==L[b,6][None,:])
        ma=U.argmax(1); ok=U.max(1)>=0.6
        for j in np.where(ok)[0]:
            pairs+=1; x,y=sm[a[j]],sl[b[ma[j]]]
            if (not x) and y: dup+=1      # M kept (outside) and L kept (inside): duplicate
            if x and (not y): lost+=1     # M dropped (inside) and L dropped (outside): lost
    print(rule,'matched confident pairs',pairs,'duplicated',dup,f'({dup/pairs:.3%})','lost',lost,f'({lost/pairs:.3%})',flush=True)
