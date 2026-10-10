import sys, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent5')
from mix_eval import ev, scores, MODELS
from tiles_common import load, tile_of, N
from anchor_tiles import anchored_tiles, iou
M=load(MODELS['M']); E=load(MODELS['L']); G=10; k=16
print('M alone', np.round(ev(M),4), flush=True)
rng=np.random.default_rng(0); sc=scores('Munc0.01',G,M,E,rng)
sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1)
tm=tile_of(M,G); te=anchored_tiles(M,E,G,'ML'); Mi=M[:,0].astype(int); Ei=E[:,0].astype(int)
inM=S[Mi,tm]; inE=S[Ei,te]
# per-image same-class IoU matching inside routed tiles
newscore=M[:,5].copy(); matched=np.zeros(len(M),bool); Ematched=np.zeros(len(E),bool)
mo=np.argsort(Mi,kind='stable'); eo=np.argsort(Ei,kind='stable'); mb=np.searchsorted(Mi[mo],np.arange(N+1)); eb=np.searchsorted(Ei[eo],np.arange(N+1))
for i in range(N):
    a=mo[mb[i]:mb[i+1]]; a=a[inM[a]]; b=eo[eb[i]:eb[i+1]]; b=b[inE[b]]
    if len(a)==0 or len(b)==0: continue
    U=iou(M[a,1:5],E[b,1:5])*(M[a,6][:,None]==E[b,6][None,:])
    # one-to-one greedy: L detections in descending score each claim their best unclaimed M box (IoU>=0.5, same class)
    U=U.copy()
    for q in np.argsort(-E[b,5]):
        c=U[:,q].argmax()
        if U[c,q]>=0.5:
            newscore[a[c]]=E[b[q],5]; matched[a[c]]=True; Ematched[b[q]]=True; U[c,:]=-1
def run(tag,arr): st=ev(arr); print(f'{tag}: AP {st[0]:.4f} S/M/L {st[1]:.4f}/{st[2]:.4f}/{st[3]:.4f}',flush=True)
run('full replacement (agent 5 row)', np.concatenate([M[~inM],E[inE]]))
R=M.copy(); R[:,5]=newscore
run('rescore only: M boxes in routed tiles take the matched L score, unmatched keep M score', R)
R2=R.copy(); R2[inM & ~matched,5]*=0.1
run('rescore only, unmatched M boxes x0.1', R2)
run('add only: M unchanged + unmatched L detections in routed tiles', np.concatenate([M,E[inE & ~Ematched]]))
run('rescore + add (M geometry for matched objects)', np.concatenate([R2,E[inE & ~Ematched]]))
print('routed-tile counts: M dets %d (matched %d), L dets %d (unmatched %d)'%(inM.sum(),matched.sum(),inE.sum(),(inE&~Ematched).sum()))
