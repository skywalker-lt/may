"""L448/L576 (and L448/L640, L448/L512) count routes at finer shares; paired bootstrap of routed minus the dense time-share
of L512/L576 at the same est. latency (cost model A, router charged), i.e. the measured part of the envelope."""
import sys, json, numpy as np
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent6/r3')
exec(open('rs2_hosts.py').read().split("rng=np.random.default_rng(0)")[0])
rng=np.random.default_rng(1); o=np.argsort(pc)
for lo,hi,shares in (('L448','L576',(0.4,0.5,0.6,0.7)),('L448','L640',(0.5,0.7)),('L448','L512',(0.5,))):
    for s in shares:
        n=int(s*nI); sel=np.full(nI,ix[hi],np.int8); sel[o[:n]]=ix[lo]; ap=M.ap(sel); t=s*ms[lo]+(1-s)*ms[hi]+0.18
        print(f'{lo}/{hi} share {s}: AP {ap:.4f} at est. {t:.2f} ms; envelope {env(t):.4f} ({ap-env(t):+.4f})',flush=True)
lo,hi='L448','L576'
for s in (0.5,0.6):
    n=int(s*nI); sel=np.full(nI,ix[hi],np.int8); sel[o[:n]]=ix[lo]; t=s*ms[lo]+(1-s)*ms[hi]+0.18
    f=(ms['L576']-t)/(ms['L576']-ms['L512'])  # share of L512 in the dense time-share at latency t
    d=[]
    for b in range(40):
        w=np.bincount(rng.integers(0,nI,nI),minlength=nI).astype(float)
        ts=np.full(nI,ix['L576'],np.int8); ts[rng.permutation(nI)[:int(f*nI)]]=ix['L512']
        d.append(M.ap(sel,None,w)-M.ap(ts,None,w))
    print(f'share {s}: routed minus dense L512/L576 time-share (L512 share {f:.2f}) at {t:.2f} ms: {np.mean(d):+.4f} sd {np.std(d):.4f} (40 paired draws)',flush=True)
print('DONE')
