"""Which resolution-route host could carry rung-private weights past the dense envelope? Two-scale count routes on public
M and L (sparsest share to the low scale, n320 out-of-fold ridge of log1p(GT count)), router 0.18 ms charged on every image,
T4 unset basis, latencies of unmeasured scales from agent 7's cost model A (est.). Envelope = upper concave hull of the dense
(model, scale) points (est. latencies), as agent 2's hull.log."""
import sys, json, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent6/r3')
from mixk import MixK
INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'; W='/data/tmp/ds-yolo/seminar5/work/agent6/'
src={'M512':W+'r3/ev_dumpml_yolo26m_512_coco.pkl','M576':W+'r3/ev_dumpml_yolo26m_576_coco.pkl','M640':W+'evm.pkl',
     'L448':W+'r3/ev_dumpml_yolo26l_448_coco.pkl','L512':W+'r3/ev_dumpml_yolo26l_512_coco.pkl','L576':W+'r3/ev_dumpml_yolo26l_576_coco.pkl','L640':W+'r3/ev_dump_yolo26l_coco.pkl'}
ms={'M448':3.12,'M512':3.78,'L448':4.01,'M576':4.53,'L512':4.86,'M608':4.93,'L544':5.32,'M640':5.36,'L576':5.82,'L640':6.89,'X640':12.41}
apd={'M448':0.4937,'M512':0.5063,'L448':0.5117,'M576':0.5203,'L512':0.5262,'M608':0.5234,'L544':0.5317,'M640':0.5261,'L576':0.5368,'L640':0.5417,'X640':0.5691}
pts=sorted((ms[k],apd[k]) for k in ms)
hull=[]
for p in pts:
    while len(hull)>=2 and (hull[-1][1]-hull[-2][1])*(p[0]-hull[-2][0])<=(p[1]-hull[-2][1])*(hull[-1][0]-hull[-2][0]): hull.pop()
    hull.append(p)
def env(t): xs,ys=zip(*hull); return float(np.interp(t,xs,ys))
F=lambda t:0.5261+0.0102*(t-5.36)
names=list(src); M=MixK([src[n] for n in names]); nI=len(M.imgIds); ix={n:j for j,n in enumerate(names)}
print('full AP '+' '.join(f'{n} {M.ap(np.full(nI,ix[n],np.int8)):.4f}' for n in names),flush=True)
gt=json.load(open(INP+'instances_val2017.json')); cnt=np.zeros(nI)
for g in gt['annotations']:
    if not g.get('iscrowd',0): cnt[M.img_index[g['image_id']]]+=1
z=np.load(INP+'val2017_stem_pooled.npz'); order=np.array([M.img_index[i] for i in z['image_id']]); X=np.zeros((nI,128)); X[order]=z['n320']
y=np.log1p(cnt); folds=np.random.RandomState(0).permutation(nI)%5; pc=np.zeros(nI)
for f in range(5):
    tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd; A=Xs[tr]
    w=np.linalg.solve(A.T@A+len(A)*np.eye(128)/100, A.T@(y[tr]-y[tr].mean())); pc[~tr]=Xs[~tr]@w+y[tr].mean()
rng=np.random.default_rng(0); o=np.argsort(pc)
for lo,hi in (('M512','M640'),('M576','M640'),('L448','L576'),('L512','L640'),('L512','L576'),('M512','L576'),('M512','L640')):
    for s in (0.3,0.5,0.7):
        n=int(s*nI); sel=np.full(nI,ix[hi],np.int8); sel[o[:n]]=ix[lo]; ap=M.ap(sel)
        nl=[]
        for d in range(3):
            r=np.full(nI,ix[hi],np.int8); r[rng.permutation(nI)[:n]]=ix[lo]; nl.append(M.ap(r))
        t=s*ms[lo]+(1-s)*ms[hi]+0.18
        print(f'{lo}/{hi} share {s}: AP {ap:.4f} at est. {t:.2f} ms; null {np.mean(nl):.4f} (+{ap-np.mean(nl):.4f}); packet bar {F(t)+0.003:.4f} ({ap-F(t)-0.003:+.4f}); envelope {env(t):.4f} ({ap-env(t):+.4f}, need +0.003)',flush=True)
print('DONE')
