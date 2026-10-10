"""Round-2 attack on agent 7 (Exchange routing): decompose {M@512, M@640, L@640} into the shrink leg (agent 1's RS-2) and
the L leg. The T4 front is the M-L chord (slope 0.0102 = (0.5417-0.5261)/(6.89-5.36)), so an L leg routed at random sits
exactly on the front; only its count-weighting can add anything. Exact mixtures with agent 7's mixlib and caches (read-only).
Also: per-tercile subset AP (exact pycocotools on image subsets) for every public dump, by thumbnail-predicted count."""
import os, sys, json, numpy as np
os.environ["OMP_NUM_THREADS"]="1"
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent7')
from mixlib import mix_ap, mix_ap_subset, img_ids
INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
ids=img_ids(); I=len(ids); ix={i:k for k,i in enumerate(ids)}
z=np.load(INP+'val2017_stem_pooled.npz'); X=np.zeros((I,128)); X[[ix[i] for i in z['image_id']]]=z['n320']
gt=json.load(open(INP+'instances_val2017.json')); cnt=np.zeros(I)
for a in gt['annotations']:
    if not a.get('iscrowd',0): cnt[ix[a['image_id']]]+=1
y=np.log1p(cnt); folds=np.random.RandomState(0).permutation(I)%5; p=np.zeros(I)
for f in range(5):
    tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd; A=Xs[tr]
    w=np.linalg.solve(A.T@A+len(A)*np.eye(128)/100, A.T@(y[tr]-y[tr].mean())); p[~tr]=Xs[~tr]@w+y[tr].mean()
F=lambda L: 0.5261+0.0102*(L-5.36); ROUTER=0.18
menu=["dumpml_yolo26m_512_coco","dumpml_yolo26m_coco","dump_yolo26l_coco"]; c=np.array([3.78,5.36,6.89])
def rep(a,tag):
    ap=mix_ap(menu,a); L=c[a].mean()+ROUTER
    print(f"{tag:58s} AP {ap:.4f} avg {L:.3f} shares {np.round(np.bincount(a,minlength=3)/I,3)} bar {F(L)+0.003:.4f} AP-bar {ap-F(L)-0.003:+.4f}",flush=True)
    return ap
o=np.argsort(p); rng=np.random.RandomState(7)
for q in (0.33,0.5):
    qL=(q*(5.36-3.78)-ROUTER)/(6.89-5.36); nL=int(qL*I); n5=int(q*I)
    a=np.ones(I,int); a[o[:n5]]=0; a[o[I-nL:]]=2; rep(a,f"exchange q={q} (count to 512, count to L)")
    b=np.ones(I,int); b[o[:n5]]=0; rep(b,f"shrink leg only q={q} (agent 1 RS-2, +router)")
    for d in range(3):
        r=b.copy(); rest=np.where(b==1)[0]; r[rng.permutation(rest)[:nL]]=2; rep(r,f"shrink q={q} + L at random on the rest, draw {d}")
    # L leg by count without the shrink leg, at the same L share (pure capacity routing at +latency)
    e=np.ones(I,int); e[o[I-nL:]]=2; rep(e,f"L leg only by count, share {qL:.3f}")
    for d in range(2):
        e=np.ones(I,int); e[rng.permutation(I)[:nL]]=2; rep(e,f"L leg only at random, share {qL:.3f}, draw {d}")
# per-tercile subset AP by predicted count
names=["dumpml_yolo26m_coco","dump_yolo26l_coco","dumpml_yolo26m_512_coco","dumpml_yolo26m_768_coco","dump_yolov12m_sdpa_coco","dump_yolo11m_coco","dump_yolo26s_coco"]
ter=np.zeros(I,int); ter[o[I//3:2*I//3]]=1; ter[o[2*I//3:]]=2
print("subset AP by thumbnail-predicted count tercile (sparse / middle / dense), all images",flush=True)
for n in names:
    v=[mix_ap_subset([n],np.zeros(I,int),ter==t) for t in range(3)]; full=mix_ap([n],np.zeros(I,int))
    print(f"  {n:28s} {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}   all {full:.4f}",flush=True)
print("DONE",flush=True)
