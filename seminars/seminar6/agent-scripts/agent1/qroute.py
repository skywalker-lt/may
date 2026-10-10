"""Query-level routing headroom from per-image dumps (seminar 6, agent 1, lens 1).
Base B emits <=300 candidates per image (one-to-one head, static TopK). A query refiner re-predicts k of them.
Proxy for a refiner of expert quality E: a routed candidate matched to an E detection (same class, IoU>=0.5,
greedy by E score) takes E's box and score; a routed candidate with no E match keeps its box and has its score
multiplied by alpha. Unrouted candidates are untouched. Nothing is added (a query cannot become a new object).
Usage: qroute.py BASE EXPERT k rule alpha [seed]   (k=300 -> every candidate; rule in unc|random|top|low|oracleE|band)
CPU only, OMP_NUM_THREADS=1."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, json, time, contextlib, io, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D='/data/tmp/ds-yolo/seminar6/inputs/'
DUMPS={'M640':D+'dumps/dump_yolo26m_coco.json','L640':D+'dumps/dump_yolo26l_coco.json','X640':D+'dumps/dump_yolo26x_coco.json',
       'L544':D+'dumps_r2/dumpml_yolo26l_544_coco.json','L512':D+'dumps_r2/dumpml_yolo26l_512_coco.json',
       'L576':D+'dumps_r2/dumpml_yolo26l_576_coco.json','M768':D+'dumps/dumpml_yolo26m_768_coco.json','M640o2m':D+'dumps_r2/dumpml_yolo26m_640_o2m_coco.json'}
C='/data/tmp/ds-yolo/seminar6/work/agent1/cache/'
with contextlib.redirect_stdout(io.StringIO()): cg=COCO(D+'dumps/instances_val2017.json')
ids=sorted(cg.getImgIds()); idx={i:k for k,i in enumerate(ids)}; N=len(ids)
def load(name):
    c=C+name+'.npy'
    if os.path.exists(c): return np.load(c)
    d=json.load(open(DUMPS[name]))
    a=np.array([[idx[x['image_id']],*x['bbox'],x['score'],x['category_id']] for x in d],float)
    o=np.lexsort((-a[:,5],a[:,0])); a=a[o]; np.save(c,a); return a
def iou(a,b):
    ax1,ay1,ax2,ay2=a[:,0],a[:,1],a[:,0]+a[:,2],a[:,1]+a[:,3]; bx1,by1,bx2,by2=b[:,0],b[:,1],b[:,0]+b[:,2],b[:,1]+b[:,3]
    iw=np.clip(np.minimum(ax2[:,None],bx2[None])-np.maximum(ax1[:,None],bx1[None]),0,None)
    ih=np.clip(np.minimum(ay2[:,None],by2[None])-np.maximum(ay1[:,None],by1[None]),0,None)
    inter=iw*ih; return inter/(a[:,2:4].prod(1)[:,None]+b[:,2:4].prod(1)[None]-inter+1e-9)
def match(B,E,bn,en):
    c=C+f'match_{bn}_{en}.npy'
    if os.path.exists(c): return np.load(c)
    Bi=B[:,0].astype(int); Ei=E[:,0].astype(int); bb=np.searchsorted(Bi,np.arange(N+1)); eb=np.searchsorted(Ei,np.arange(N+1))
    m=-np.ones(len(B),int)
    for i in range(N):
        A=np.arange(bb[i],bb[i+1]); Q=np.arange(eb[i],eb[i+1])
        if len(A)==0 or len(Q)==0: continue
        U=iou(B[A,1:5],E[Q,1:5])*(B[A,6][:,None]==E[Q,6][None,:])
        for q in np.argsort(-E[Q,5]):
            c_=U[:,q].argmax()
            if U[c_,q]>=0.5: m[A[c_]]=Q[q]; U[c_,:]=-1
    np.save(c,m); return m
def ev(a):
    with contextlib.redirect_stdout(io.StringIO()):
        cd=cg.loadRes(np.column_stack([np.array(ids)[a[:,0].astype(int)],a[:,1:7]]))
        e=COCOeval(cg,cd,'bbox'); e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[[0,1,2,3,4,5]]
if __name__=='__main__':
    bn,en,k,rule,alpha=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4],float(sys.argv[5]); seed=int(sys.argv[6]) if len(sys.argv)>6 else 0
    t=time.time(); B=load(bn)
    if en=='-':
        s=ev(B); print(f'ALONE {bn} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S/M/L {s[3]:.4f}/{s[4]:.4f}/{s[5]:.4f} ({time.time()-t:.0f}s)',flush=True); sys.exit()
    E=load(en); m=match(B,E,bn,en); Bi=B[:,0].astype(int); s=B[:,5]; rng=np.random.default_rng(seed)
    ok=m>=0; Rn=np.zeros(len(B)); Rn[ok]=E[m[ok],5]
    if rule=='unc': sc=s*(1-s)
    elif rule=='band': sc=-np.abs(np.log(s/(1-s)+1e-9)-np.log(0.25/0.75))   # closest to score 0.25
    elif rule=='random': sc=rng.random(len(B))
    elif rule=='slot':  # the programme's share-matched null: k random slots of the engine's 300; only scored slots (rank<n_i) exist in the dump
        sc=np.zeros(len(B)); bb0=np.searchsorted(Bi,np.arange(N+1))
        for i in range(N):
            n=bb0[i+1]-bb0[i]
            if n==0: continue
            pick=rng.choice(300,size=min(k,300),replace=False); pick=pick[pick<n]; sc[bb0[i]:bb0[i+1]]=-1; sc[bb0[i]+pick]=1
    elif rule=='top': sc=s
    elif rule=='low': sc=-s
    elif rule=='oracleE':  # expert-knowledge bound: candidates whose refinement changes the most (not GT-free)
        d=np.abs(np.where(ok,Rn,alpha*s)-s); g=np.zeros(len(B)); g[ok]=1-np.diag(iou(B[ok][:,1:5],E[m[ok]][:,1:5])) if ok.sum()<1 else 0
        # cheap per-row IoU without the full matrix
        bb=B[ok,1:5]; ee=E[m[ok],1:5]; ix=np.clip(np.minimum(bb[:,0]+bb[:,2],ee[:,0]+ee[:,2])-np.maximum(bb[:,0],ee[:,0]),0,None); iy=np.clip(np.minimum(bb[:,1]+bb[:,3],ee[:,1]+ee[:,3])-np.maximum(bb[:,1],ee[:,1]),0,None)
        inter=ix*iy; g[ok]=1-inter/(bb[:,2]*bb[:,3]+ee[:,2]*ee[:,3]-inter+1e-9); sc=d+0.5*g*np.maximum(s,Rn)
    else: raise ValueError(rule)
    # rank within image, keep top k
    bb_=np.searchsorted(Bi,np.arange(N+1)); routed=np.zeros(len(B),bool)
    for i in range(N):
        A=np.arange(bb_[i],bb_[i+1])
        if len(A)==0: continue
        routed[A[np.argsort(-sc[A])[:k]]]=True
    R=B.copy(); rr=routed&ok
    if os.environ.get('RESCORE','0')!='1': R[rr,1:5]=E[m[rr],1:5]
    R[rr,5]=E[m[rr],5]; R[routed&~ok,5]*=alpha
    if rule=='slot': routed=sc>0
    st=ev(R)
    # GT-free diagnostics: how much of the expert's total score change the routed set captures
    dall=np.abs(np.where(ok,Rn,alpha*s)-s); cap=dall[routed].sum()/dall.sum()
    print(f'QR{"-rescore" if os.environ.get("RESCORE","0")=="1" else ""} base={bn} exp={en} k={k} rule={rule} alpha={alpha} seed={seed} AP {st[0]:.4f} AP50 {st[1]:.4f} AP75 {st[2]:.4f} S/M/L {st[3]:.4f}/{st[4]:.4f}/{st[5]:.4f} | routed {routed.mean():.3f} matched-of-routed {ok[routed].mean():.3f} dScore-mass captured {cap:.3f} ({time.time()-t:.0f}s)',flush=True)
