"""Round-2 attack on agent 5 (TCR): the tile proxy replaced ALL of M's detections in a routed tile by E's; TCR replaces
only M's P3-scale anchors and keeps P4/P5. Re-score the proxy with replacement restricted to boxes whose letterbox size
sqrt(w*h)*r < thr (a P3-scale proxy), for both M's removed and E's inserted detections. Reuses agent 5's code and caches
read-only (PYTHONDONTWRITEBYTECODE=1)."""
import sys, os
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent5')
import numpy as np
from tiles_common import *
from mix_eval import ev, scores, MODELS, load
from anchor_tiles import anchored_tiles
base,exp,G,share,rule,thr=sys.argv[1],sys.argv[2],int(sys.argv[3]),float(sys.argv[4]),sys.argv[5],float(sys.argv[6])
seed=int(sys.argv[7]) if len(sys.argv)>7 else 0
M=load(MODELS[base]); E=load(MODELS[exp]); rng=np.random.default_rng(seed)
k=int(round(share*G*G)); sc=scores(rule,G,M,E,rng)
sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1)
def small(a):
    i=a[:,0].astype(int); return np.sqrt(a[:,3]*a[:,4])*R[i] < thr
inM=S[M[:,0].astype(int),tile_of(M,G)]; inE=S[E[:,0].astype(int),anchored_tiles(M,E,G,base+exp)]
if thr>0: dropM=inM&small(M); addE=inE&small(E)
else: dropM=inM; addE=inE
a=np.concatenate([M[~dropM],E[addE]]); st=ev(a)
print(f'P3ONLY base={base} exp={exp} share={share:.2f} rule={rule} thr={thr:g} seed={seed} AP={st[0]:.4f} S/M/L={st[1]:.4f}/{st[2]:.4f}/{st[3]:.4f}',flush=True)
