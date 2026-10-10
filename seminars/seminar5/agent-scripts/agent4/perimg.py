# Per-image AP proxy for each dump (pycocotools evalImgs, area 'all', maxDet 100) + overall AP.
import os, sys, json, time, numpy as np, contextlib, io
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D='/data/tmp/ds-yolo/seminar5/inputs/dumps/'; W='/data/tmp/ds-yolo/seminar5/work/agent4/'
with contextlib.redirect_stdout(io.StringIO()): gt=COCO(D+'instances_val2017.json')
imgIds=sorted(gt.getImgIds()); catIds=sorted(gt.getCatIds())
rs=np.linspace(0,1,101)
def perimg(E):
    I=len(E.params.imgIds); A=len(E.params.areaRng); K=len(E.params.catIds)
    ap=np.zeros(I); n=np.zeros(I)
    for k in range(K):
        for i in range(I):
            e=E.evalImgs[k*A*I+0*I+i]
            if e is None: continue
            gi=np.array(e['gtIgnore'],bool); npig=(~gi).sum()
            if npig==0: continue
            dm=np.array(e['dtMatches'])[:, :100]; di=np.array(e['dtIgnore'],bool)[:, :100]
            s=0.
            for t in range(dm.shape[0]):
                keep=~di[t]; m=dm[t][keep]
                tp=np.cumsum(m>0); fp=np.cumsum(m==0)
                if len(m)==0: continue
                rc=tp/npig; pr=tp/np.maximum(tp+fp,1e-9)
                pr=np.maximum.accumulate(pr[::-1])[::-1]
                ix=np.searchsorted(rc,rs,side='left'); q=np.where(ix<len(pr),pr[np.minimum(ix,len(pr)-1)],0.)
                s+=q.mean()
            ap[i]+=s/dm.shape[0]; n[i]+=1
    return np.where(n>0,ap/np.maximum(n,1),np.nan)
def run(name, dets):
    t=time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(dets); E=COCOeval(gt,dt,'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    print(f'{name} AP={E.stats[0]:.4f} S/M/L={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f} ({time.time()-t:.0f}s)',flush=True)
    return E
if __name__=='__main__':
    for name,f in [x.split('=') for x in sys.argv[1:]]:
        E=run(name,D+f); p=perimg(E)
        np.savez(W+f'pi_{name}.npz',imgIds=np.array(E.params.imgIds),ap=p,stat=np.array(E.stats))
        print(name,'per-image mean',np.nanmean(p),flush=True)
