"""Score forced-branch dumps of a routed weight-bank arm (agent 6, round 3).
usage: score_forced.py --experts n0=dump0.json,... --route route.json [--extra merged=..,dense=..] [--boot 100] [--pairs]
Sources may be COCO json dumps (an evalImgs pickle is cached next to this script) or ready pickles (.pkl).
Top-1: route value [c]; pair arm: route value [i,j] -> expert names must be pair names 'pairij' in --experts order."""
import sys, os, json, argparse, itertools, subprocess, numpy as np, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mixk import MixK
W=os.path.dirname(os.path.abspath(__file__))+'/'; INP='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
PREP='/data/tmp/ds-yolo/seminar5/work/agent6/prep_eval.py'
ap_=argparse.ArgumentParser(); ap_.add_argument('--experts',required=True); ap_.add_argument('--route',default=None)
ap_.add_argument('--route-npy',default=None); ap_.add_argument('--extra',default=''); ap_.add_argument('--boot',type=int,default=100)
ap_.add_argument('--pairs',action='store_true'); ap_.add_argument('--tag',default='run')
a=ap_.parse_args(); t0=time.time()
def P(*x): print(*x,flush=True)
def pkl(path):
    if path.endswith('.pkl'): return path
    out=W+'ev_'+os.path.basename(path).replace('.json','.pkl')
    if not os.path.exists(out):
        subprocess.run(['/data/envs/rtdetr/bin/python',PREP,path,out],check=True,stdout=subprocess.DEVNULL)
    return out
ex=[x.split('=',1) for x in a.experts.split(',')]; xt=[x.split('=',1) for x in a.extra.split(',') if x]
names=[n for n,_ in ex]+[n for n,_ in xt]; E=len(ex)
M=MixK([pkl(p) for _,p in ex+xt]); nI=len(M.imgIds); P(f'[{a.tag}] loaded {len(names)} sources in {time.time()-t0:.0f}s')
# route -> branch index per image
if a.route:
    r=json.load(open(a.route)); route=np.full(nI,-1)
    if a.pairs:
        pid={tuple(sorted(int(c) for c in n[-2:])):j for j,n in enumerate(names[:E])}
        for k,v in r.items(): route[M.img_index[int(k)]]=pid[tuple(sorted(v))]
    else:
        for k,v in r.items(): route[M.img_index[int(k)]]=v[0]
else: route=np.load(a.route_npy)
assert (route>=0).all(), 'images without a route'
gt=json.load(open(INP+'instances_val2017.json')); cnt=np.zeros(nI)
for g in gt['annotations']:
    if not g.get('iscrowd',0): cnt[M.img_index[g['image_id']]]+=1
full={n:M.ap(np.full(nI,j,np.int8)) for j,n in enumerate(names)}
P('full-val AP: '+'  '.join(f'{n} {v:.4f}' for n,v in full.items()))
S=[route==c for c in range(E)]
P('route shares: '+'  '.join(f'{names[c]} {S[c].mean():.3f} (mean GT count {cnt[S[c]].mean():.2f})' for c in range(E)))
mat=np.array([[M.ap(np.full(nI,b,np.int8),S[c]) if S[c].any() else np.nan for c in range(E)] for b in range(len(names))])
P('subset AP, rows = forced source, columns = images routed to branch c'); P('            '+''.join(f'{names[c]:>12s}' for c in range(E)))
for b in range(len(names)): P(f'{names[b]:>12s}'+''.join(f'{mat[b,c]:12.4f}' for c in range(E)))
dmean=np.array([mat[c,c]-np.nanmean(np.delete(mat[:E,c],c)) for c in range(E)]); dmax=np.array([mat[c,c]-np.nanmax(np.delete(mat[:E,c],c)) for c in range(E)])
P('diagonal minus mean of the other experts, per column: '+' '.join(f'{x:+.4f}' for x in dmean)+f'   mean {np.nanmean(dmean):+.4f}')
P('diagonal minus best other expert, per column:         '+' '.join(f'{x:+.4f}' for x in dmax)+f'   mean {np.nanmean(dmax):+.4f}')
if E+1<=len(names):
    for j in range(E,len(names)): P(f'diagonal minus {names[j]}, per column: '+' '.join(f'{mat[c,c]-mat[j,c]:+.4f}' for c in range(E)))
mm=mat[:E,:E]; dc=mm-mm.mean(1,keepdims=True)-mm.mean(0,keepdims=True)+mm.mean()
P('interaction (double-centred) diagonal, per column: '+' '.join(f'{dc[c,c]:+.4f}' for c in range(E))+f'   mean {np.mean(np.diag(dc)):+.4f}  (off-diagonal mean {(dc.sum()-np.trace(dc))/(E*E-E):+.4f})')
sel=route.astype(np.int8); rout=M.ap(sel); best=max(full[n] for n in names[:E])
P(f'routed AP {rout:.4f}; best single forced expert {best:.4f} ({rout-best:+.4f}); '+' '.join(f'{n} {full[n]:.4f} ({rout-full[n]:+.4f})' for n in names[E:]))
# branch-permuted nulls: relabel experts (top-1: all permutations of E; pairs: permutations of the 4 experts induce pair relabels)
perm=[]
if not a.pairs:
    for p in itertools.permutations(range(E)):
        if p==tuple(range(E)): continue
        perm.append(M.ap(np.array(p,np.int8)[route]))
else:
    pn=[tuple(sorted(int(c) for c in n[-2:])) for n in names[:E]]; pid={x:j for j,x in enumerate(pn)}
    for p in itertools.permutations(range(4)):
        if p==(0,1,2,3): continue
        mp=np.array([pid[tuple(sorted((p[i],p[j])))] for (i,j) in pn],np.int8); perm.append(M.ap(mp[route]))
perm=np.array(perm); P(f'branch-permuted null: mean {perm.mean():.4f} sd {perm.std():.4f} max {perm.max():.4f} over {len(perm)} relabels; routed - mean {rout-perm.mean():+.4f}')
rng=np.random.default_rng(0); rnd=[]
for d in range(5): rnd.append(M.ap(rng.permutation(route).astype(np.int8)))
P(f'random route at the same shares (5 draws): mean {np.mean(rnd):.4f} sd {np.std(rnd):.4f}; routed - mean {rout-np.mean(rnd):+.4f}')
# count terciles: does any expert specialise along object count?
z=np.load(INP+'val2017_stem_pooled.npz'); order=np.array([M.img_index[i] for i in z['image_id']]); X=np.zeros((nI,128)); X[order]=z['n320']
y=np.log1p(cnt); folds=np.random.RandomState(0).permutation(nI)%5; pc=np.zeros(nI)
for f in range(5):
    tr=folds!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xs=(X-mu)/sd; A=Xs[tr]
    w=np.linalg.solve(A.T@A+len(A)*np.eye(128)/100, A.T@(y[tr]-y[tr].mean())); pc[~tr]=Xs[~tr]@w+y[tr].mean()
q=np.quantile(pc,[1/3,2/3]); T3=[pc<q[0],(pc>=q[0])&(pc<q[1]),pc>=q[1]]
P('subset AP by thumbnail-predicted count tercile (sparse / middle / dense):')
for b in range(len(names)): P(f'{names[b]:>12s} '+' '.join(f'{M.ap(np.full(nI,b,np.int8),t):.4f}' for t in T3))
P('route share by tercile: '+' | '.join(' '.join(f'{(route[t]==c).mean():.2f}' for c in range(E)) for t in T3))
# paired bootstrap over images
if a.boot:
    bd=[];br=[];bm=[];bi=[]
    for i in range(a.boot):
        wv=np.bincount(rng.integers(0,nI,nI),minlength=nI).astype(float)
        m2=np.array([[M.ap(np.full(nI,b,np.int8),S[c],wv) for c in range(E)] for b in range(E)])
        bd.append(np.nanmean([m2[c,c]-np.nanmean(np.delete(m2[:,c],c)) for c in range(E)]))
        d2=m2-m2.mean(1,keepdims=True)-m2.mean(0,keepdims=True)+m2.mean(); bi.append(np.mean(np.diag(d2)))
        rr=M.ap(sel,None,wv); bs=[M.ap(np.full(nI,b,np.int8),None,wv) for b in range(E)]
        br.append(rr-bs[int(np.argmax([full[n] for n in names[:E]]))])
        if E<len(names): bm.append(rr-M.ap(np.full(nI,E,np.int8),None,wv))
        if (i+1)%25==0: P(f'  boot {i+1}/{a.boot}  ({time.time()-t0:.0f}s)')
    P(f'bootstrap ({a.boot}): mean diagonal advantage {np.mean(bd):+.4f} sd {np.std(bd):.4f}; interaction diagonal {np.mean(bi):+.4f} sd {np.std(bi):.4f}; routed - best single {np.mean(br):+.4f} sd {np.std(br):.4f}'+(f'; routed - {names[E]} {np.mean(bm):+.4f} sd {np.std(bm):.4f}' if bm else ''))
P(f'DONE {time.time()-t0:.0f}s')
