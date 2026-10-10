"""Round 2: P3-scale-restricted tile mixture. In the k routed tiles (M's uncertainty mass, as round 1), only detections
whose letterbox size sqrt(w*h) < thr are swapped (M's out, E's in); larger detections stay M's everywhere, as in TCR,
which replaces only M's stride-8 anchors. E detections are tiled and sized by their best same-class M match (IoU>=0.5)
when one exists. Single-label pairs (yolo11m -> yolov12m) also supported. Scored with pycocotools."""
import sys, os, numpy as np
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent5')
from tiles_common import *
from mix_eval import ev, scores, MODELS
from anchor_tiles import iou
MODELS.update({'Y11':'dump_yolo11m_coco.json','Y12':'dump_yolov12m_sdpa_coco.json'})
def match(M,E,tag,G=10):
    f=f'/data/tmp/ds-yolo/seminar5/work/agent5/r2/cache_match_{tag}.npy'
    if os.path.exists(f): return np.load(f)
    m=-np.ones(len(E),int); Mi=M[:,0].astype(int); Ei=E[:,0].astype(int)
    mo=np.argsort(Mi,kind='stable'); eo=np.argsort(Ei,kind='stable')
    mb=np.searchsorted(Mi[mo],np.arange(N+1)); eb=np.searchsorted(Ei[eo],np.arange(N+1))
    for i in range(N):
        a=mo[mb[i]:mb[i+1]]; b=eo[eb[i]:eb[i+1]]
        if len(a)==0 or len(b)==0: continue
        U=iou(E[b,1:5],M[a,1:5])*(E[b,6][:,None]==M[a,6][None,:])
        j=U.argmax(1); ok=U.max(1)>=0.5; m[b[ok]]=a[j[ok]]
    np.save(f,m); return m
def lsize(a): i=a[:,0].astype(int); return np.sqrt(a[:,3]*a[:,4])*R[i]
if __name__=='__main__':
    base,exp,share,rule,thr=sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4],float(sys.argv[5])
    seed=int(sys.argv[6]) if len(sys.argv)>6 else 0; G=10
    M=load(MODELS[base]); E=load(MODELS[exp]); rng=np.random.default_rng(seed)
    k=int(round(share*G*G)); sc=scores(rule,G,M,E,rng)
    sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1)
    mm=match(M,E,base+exp); tm=tile_of(M,G); te=tile_of(E,G).copy(); se=lsize(E); sm=lsize(M)
    ok=mm>=0; te[ok]=tm[mm[ok]]; se[ok]=sm[mm[ok]]
    outM=S[M[:,0].astype(int),tm]&(sm<thr); inE=S[E[:,0].astype(int),te]&(se<thr)
    st=ev(np.concatenate([M[~outM],E[inE]]))
    print(f'P3MIX base={base} exp={exp} share={share:.2f} rule={rule} thr={thr:g} seed={seed} swappedM={outM.sum()} insertedE={inE.sum()} AP={st[0]:.4f} S/M/L={st[1]:.4f}/{st[2]:.4f}/{st[3]:.4f}', flush=True)
