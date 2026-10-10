"""Round-2 attack on direction 5 (TCR): the tile expert only replaces P3-scale anchors (stride 8 + stride-4 sub-anchors);
M's P4/P5 anchors stay. So the proxy should swap in E's detections only for boxes at P3 scale inside the routed tiles.
Size = sqrt(w*h) at the 640 letterbox scale. Reuses agent 5's tile code read-only (imports only; caches already exist)."""
import sys, os
os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent5')
import numpy as np
from mix_eval import *   # ev, scores, MODELS, load, tile_of, anchored_tiles, N, R
base,exp,G,share,rule,thr=sys.argv[1],sys.argv[2],int(sys.argv[3]),float(sys.argv[4]),sys.argv[5],float(sys.argv[6])
seed=int(sys.argv[7]) if len(sys.argv)>7 else 0
M=load(MODELS[base]); E=load(MODELS[exp]); rng=np.random.default_rng(seed)
k=int(round(share*G*G)); sc=scores(rule,G,M,E,rng)
sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1)
def size(a): i=a[:,0].astype(int); return np.sqrt(a[:,3]*a[:,4])*R[i]
inM=S[M[:,0].astype(int),tile_of(M,G)]; inE=S[E[:,0].astype(int),anchored_tiles(M,E,G,base+exp)]
small_M=size(M)<thr; small_E=size(E)<thr
a=np.concatenate([M[~inM], M[inM & ~small_M], E[inE & small_E]])
st=ev(a)
print(f'SCALEMIX base={base} exp={exp} G={G} share={share:.2f} rule={rule} thr={thr:.0f} seed={seed} AP={st[0]:.4f} S/M/L={st[1]:.4f}/{st[2]:.4f}/{st[3]:.4f}',flush=True)
