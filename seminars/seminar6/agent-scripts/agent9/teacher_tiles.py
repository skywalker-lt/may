"""Agent 9, seminar 6, round 1. Per-tile teacher bounds for a routed-distillation construct on the tile-choice route.
Route: TCR-3's parameter-free rule (uncertainty mass sum s(1-s) of the base's own dets, score>=0.01, 10x10 letterbox tiles,
16 kept per image). Everything here is a ceiling on what a per-tile distilled expert could reach if it reproduced the
teacher's detections inside the routed tiles (every anchor, object-anchored tile assignment as in agent 5's round 3).
Reads agent 5's tile helpers (read-only) and the seminar dumps; writes caches only under work/agent9/.
ONE thread; each pycocotools evaluation is about a minute."""
import os; os.environ['OMP_NUM_THREADS']='1'
import sys, json, time, contextlib, io, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent5')
from tiles_common import tile_of, gt_array, N, R, ids, idx, D as D5
from anchor_tiles import iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
W9='/data/tmp/ds-yolo/seminar6/work/agent9/'
D6='/data/tmp/ds-yolo/seminar6/inputs/'
with contextlib.redirect_stdout(io.StringIO()): cg=COCO(D5+'instances_val2017.json')
gt=gt_array(); gt=gt[gt[:,7]==0]; gi=gt[:,0].astype(int)
SRC={'M':D5+'dump_yolo26m_coco.json','L':D5+'dump_yolo26l_coco.json','X':D5+'dump_yolo26x_coco.json',
     'M768':D5+'dumpml_yolo26m_768_coco.json','L512':D6+'dumps_r2/dumpml_yolo26l_512_coco.json'}
def load(name):
    c=W9+f'cache_{name}.npy'
    if os.path.exists(c): return np.load(c)
    d=json.load(open(SRC[name]))
    a=np.array([[idx[x['image_id']],*x['bbox'],x['score'],x['category_id']] for x in d],float); np.save(c,a); return a
def ev(a):
    with contextlib.redirect_stdout(io.StringIO()):
        cd=cg.loadRes(np.column_stack([np.array(ids)[a[:,0].astype(int)],a[:,1:7]]))
        e=COCOeval(cg,cd,'bbox'); e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[[0,1,2,3,4,5]]
def run(tag,a):
    t=time.time(); s=ev(a)
    print(f'{tag:78s} AP {s[0]:.4f} AP50 {s[1]:.4f} AP75 {s[2]:.4f} S/M/L {s[3]:.4f}/{s[4]:.4f}/{s[5]:.4f}  ({time.time()-t:.0f}s)',flush=True)
    return s
def anchored(B,E,G,tag):
    """E's dets take the tile of their best same-class match in the base B (IoU>=0.5), else their own centre tile."""
    f=W9+f'cache_anch_{tag}_{G}.npy'
    if os.path.exists(f): return np.load(f)
    tb=tile_of(B,G); te=tile_of(E,G).copy(); Bi=B[:,0].astype(int); Ei=E[:,0].astype(int)
    bo=np.argsort(Bi,kind='stable'); eo=np.argsort(Ei,kind='stable')
    bb=np.searchsorted(Bi[bo],np.arange(N+1)); eb=np.searchsorted(Ei[eo],np.arange(N+1))
    for i in range(N):
        a=bo[bb[i]:bb[i+1]]; b=eo[eb[i]:eb[i+1]]
        if len(a)==0 or len(b)==0: continue
        U=iou(E[b,1:5],B[a,1:5])*(E[b,6][:,None]==B[a,6][None,:]); j=U.argmax(1); ok=U.max(1)>=0.5; te[b[ok]]=tb[a[j[ok]]]
    np.save(f,te); return te
def unc_route(B,G,k,rng):
    tb=tile_of(B,G); s=B[:,5]>=0.01; c=np.zeros((N,G*G)); np.add.at(c,(B[s,0].astype(int),tb[s]),B[s,5]*(1-B[s,5]))
    sc=c+1e-6*rng.random((N,G*G)); sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1); return S
def rand_route(G,k,rng):
    sel=np.argsort(-rng.random((N,G*G)),1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1); return S
def quality(A,thr=0.5):
    """per-GT quality: best same-class det score with IoU>=thr (0 if none)"""
    q=np.zeros(len(gt)); Ai=A[:,0].astype(int)
    ao=np.argsort(Ai,kind='stable'); go=np.argsort(gi,kind='stable')
    ab=np.searchsorted(Ai[ao],np.arange(N+1)); gb=np.searchsorted(gi[go],np.arange(N+1))
    for i in range(N):
        a=ao[ab[i]:ab[i+1]]; g=go[gb[i]:gb[i+1]]
        if len(a)==0 or len(g)==0: continue
        U=iou(gt[g,1:5],A[a,1:5])*(gt[g,6][:,None]==A[a,6][None,:]); q[g]=np.where(U>=thr,A[a,5][None,:],0).max(1)
    return q
def tile_gain(qb,qe,G):
    """per (image, tile) net quality gain of E over the base, GT objects anchored by their centre tile"""
    tg=tile_of(gt,G); c=np.zeros((N,G*G)); np.add.at(c,(gi,tg),qe-qb); return c
def mix(B,S,E,te):
    Bi=B[:,0].astype(int); tb=tile_of(B,G); Ei=E[:,0].astype(int)
    return np.concatenate([B[~S[Bi,tb]], E[S[Ei,te]]])
G=10; k=16; rng=np.random.default_rng(0)
part=sys.argv[1] if len(sys.argv)>1 else 'none'
if part in ('A','all'):
    print('== A. base M, uncertainty route (TCR-3), 16 of 100 tiles, teachers replace every anchor in routed tiles',flush=True)
    M=load('M'); S=unc_route(M,G,k,rng); Mi=M[:,0].astype(int); tm=tile_of(M,G)
    T={n:load(n) for n in ('L','X','M768')}; te={n:anchored(M,T[n],G,'M'+n) for n in T}
    qm=quality(M); q={n:quality(T[n]) for n in T}; g={n:tile_gain(qm,q[n],G) for n in T}
    tg=tile_of(gt,G); ins=S[gi,tg]
    for n in T: print(f'   share of {n} net gain (IoU>=0.5 quality) inside the routed tiles: {(q[n]-qm)[ins].sum()/(q[n]-qm).sum():.3f}; net gain sum {(q[n]-qm).sum():.1f}',flush=True)
    names=['L','X','M768']; Gs=np.stack([g[n] for n in names],-1)          # N x 100 x 3
    best=Gs.argmax(-1); bestgain=Gs.max(-1)
    sh=[(best[S]==j).mean() for j in range(3)]; print(f'   per-tile GT-oracle teacher shares in routed tiles (L / X / M768): {sh[0]:.2f} / {sh[1]:.2f} / {sh[2]:.2f}; tiles where no teacher gains: {(bestgain[S]<=0).mean():.2f}',flush=True)
    # mixture-of-teachers oracle: in each routed tile take the teacher with the largest GT gain (ties -> X)
    rows=[M[~S[Mi,tm]]]
    for j,n in enumerate(names):
        Sj=S&(best==j); E=T[n]; Ei=E[:,0].astype(int); rows.append(E[Sj[Ei,te[n]]])
    run('A1 per-tile oracle over teachers {L, X, M768} in the routed tiles (ceiling for a mixture of teachers)',np.concatenate(rows))
    # oracle that may also keep M (a teacher only where it gains)
    keep=S&(bestgain<=0); rows=[M[~S[Mi,tm]|keep[Mi,tm]]]
    for j,n in enumerate(names):
        Sj=S&(best==j)&(bestgain>0); E=T[n]; Ei=E[:,0].astype(int); rows.append(E[Sj[Ei,te[n]]])
    run('A2 same oracle, M kept where no teacher gains',np.concatenate(rows))
    # X-only re-check in the same code path (agent 5 reports 0.5643)
    run('A3 X every anchor in the routed tiles (check against agent 5: 0.5643)',mix(M,S,T['X'],te['X']))
if part in ('B','all'):
    print('== B. base L@512 (shrink-and-refine), uncertainty route of L@512 itself, 16 of 100 tiles',flush=True)
    B=load('L512'); S=unc_route(B,G,k,rng); Sr=rand_route(G,k,rng)
    T={n:load(n) for n in ('L','X')}; te={n:anchored(B,T[n],G,'L512'+n) for n in T}
    qb=quality(B); tg=tile_of(gt,G); ins=S[gi,tg]; insr=Sr[gi,tg]
    for n in T:
        d=quality(T[n])-qb; print(f'   share of {n}@640 net gain over L@512 inside the routed tiles: {d[ins].sum()/d.sum():.3f} (random tiles {d[insr].sum()/d.sum():.3f}); net gain sum {d.sum():.1f}',flush=True)
    run('B0 L@512 alone (check: 0.5262)',B)
    run('B1 L@640 every anchor in the routed tiles of L@512 (ceiling at L quality)',mix(B,S,T['L'],te['L']))
    run('B2 X@640 every anchor in the routed tiles of L@512 (ceiling at X quality)',mix(B,S,T['X'],te['X']))
    run('B3 X@640 in 16 random tiles of L@512 (tile null at X quality)',mix(B,Sr,T['X'],te['X']))
print('DONE')
