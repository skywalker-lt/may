"""Is the +0.0026-0.0040 count-router lift of round 1 (same-cost pairs, sparse images to the weaker model) specialisation
or instance weighting? Build a weaker twin of YOLO26-M whose degradation is image-independent by construction (each
detection's logit moved by N(0, sigma), seed 0), then route it exactly as in round 1 (n320 / m640 out-of-fold count
ridge, sparsest share to B) against the share null. A twin has zero specialisation, so whatever lift it gets is pure
instance weighting. Compare lift / gap with the YOLOv12-M and YOLO11-M pairs."""
import sys, json, numpy as np, subprocess, os, pickle
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent6')
W='/data/tmp/ds-yolo/seminar5/work/agent6/'; INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
sig=float(sys.argv[1]); tag=f'twin_s{sig:g}'
if not os.path.exists(W+f'r2/ev_{tag}.pkl'):
    d=json.load(open(INP+'dumpml_yolo26m_coco.json')); rng=np.random.default_rng(0)
    s=np.clip(np.array([x['score'] for x in d]),1e-6,1-1e-6); z=np.log(s/(1-s))+rng.normal(0,sig,len(s)); s2=1/(1+np.exp(-z))
    for x,v in zip(d,s2): x['score']=float(v)
    json.dump(d,open(W+f'r2/dump_{tag}.json','w'))
    subprocess.run(['/data/envs/rtdetr/bin/python',W+'prep_eval.py',W+f'r2/dump_{tag}.json',W+f'r2/ev_{tag}.pkl'],check=True)
    os.remove(W+f'r2/dump_{tag}.json')
from mixacc import Mix
M=Mix(W+'evm.pkl',W+f'r2/ev_{tag}.pkl'); nI=len(M.imgIds)
z=np.load(INP+'val2017_stem_pooled.npz'); order=np.array([M.img_index[i] for i in z['image_id']])
N320=np.zeros((nI,128)); N320[order]=z['n320']; M640=np.zeros((nI,512)); M640[order]=z['m640']
gt=json.load(open(INP+'instances_val2017.json')); cnt=np.zeros(nI)
for a in gt['annotations']:
    if not a.get('iscrowd',0): cnt[M.img_index[a['image_id']]]+=1
y=np.log1p(cnt); rng=np.random.RandomState(0); folds=rng.permutation(nI)%5
def ridge_oof(X,y,lam=1.0):
    p=np.zeros(nI)
    for f in range(5):
        tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd; A_=Xs[tr]
        w=np.linalg.solve(A_.T@A_+lam*len(A_)*np.eye(X.shape[1])/100, A_.T@(y[tr]-y[tr].mean())); p[~tr]=Xs[~tr]@w+y[tr].mean()
    return p
A=M.ap(np.zeros(nI,np.int8)); B=M.ap(np.ones(nI,np.int8)); print(f'{tag}: A (M) {A:.4f}  B (twin) {B:.4f}  gap {A-B:.4f}',flush=True)
for nm,X in (('n320',N320),('m640',M640)):
    p=ridge_oof(X,y)
    for share in (0.25,0.5):
        n=int(share*nI); s=np.zeros(nI,np.int8); s[np.argsort(p)[:n]]=1
        nulls=[]
        for d in range(5):
            r=np.zeros(nI,np.int8); r[rng.permutation(nI)[:n]]=1; nulls.append(M.ap(r))
        ap=M.ap(s); inst=cnt[s==1].sum()/cnt.sum()
        print(f'  {nm} share {share}: count-routed {ap:.4f} null {np.mean(nulls):.4f} (sd {np.std(nulls):.4f}) lift {ap-np.mean(nulls):+.4f} lift/gap {(ap-np.mean(nulls))/(A-B):.3f} instance share routed {inst:.3f}',flush=True)
print('DONE',flush=True)
