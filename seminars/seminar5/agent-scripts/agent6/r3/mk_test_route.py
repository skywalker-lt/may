# test route for the scorer: count quartiles of the n320 out-of-fold ridge -> 4 sources (public proxies)
import numpy as np, json, pickle
INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
D=pickle.load(open('/data/tmp/ds-yolo/seminar5/work/agent6/evm.pkl','rb')); ids=D['imgIds']; ix={i:j for j,i in enumerate(ids)}; nI=len(ids)
gt=json.load(open(INP+'instances_val2017.json')); cnt=np.zeros(nI)
for g in gt['annotations']:
    if not g.get('iscrowd',0): cnt[ix[g['image_id']]]+=1
z=np.load(INP+'val2017_stem_pooled.npz'); order=np.array([ix[i] for i in z['image_id']]); X=np.zeros((nI,128)); X[order]=z['n320']
y=np.log1p(cnt); folds=np.random.RandomState(0).permutation(nI)%5; pc=np.zeros(nI)
for f in range(5):
    tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd; A=Xs[tr]
    w=np.linalg.solve(A.T@A+len(A)*np.eye(128)/100, A.T@(y[tr]-y[tr].mean())); pc[~tr]=Xs[~tr]@w+y[tr].mean()
q=np.quantile(pc,[.25,.5,.75]); r=np.digitize(pc,q); np.save('test_route.npy',r); print(np.bincount(r))
