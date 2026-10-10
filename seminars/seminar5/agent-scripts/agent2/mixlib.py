import os, pickle, json, io, contextlib, numpy as np
os.environ.setdefault('OMP_NUM_THREADS','1')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D='/data/tmp/ds-yolo/seminar5/inputs/dumps'; C='/data/tmp/ds-yolo/seminar5/work/agent2/cache'
_gt=None
def gt():
    global _gt
    if _gt is None:
        with contextlib.redirect_stdout(io.StringIO()): _gt=COCO(f'{D}/instances_val2017.json')
    return _gt
def load(name): return pickle.load(open(f'{C}/{name}.pkl','rb'))
def mix_ap(caches, choice):
    """caches: list of cached evals; choice: int array len(imgIds) selecting cache per image. Exact COCO AP of the mixture."""
    c0=caches[0]; I=len(c0['imgIds']); n=len(c0['evalImgs']); idx=np.arange(n)%I
    sel=np.asarray(choice)[idx]
    ev=[caches[s]['evalImgs'][j] for j,s in enumerate(sel)]
    g=gt(); E=COCOeval(g,None,'bbox'); E.params.imgIds=c0['imgIds']; E.params.catIds=c0['catIds']
    E.evalImgs=ev; E._paramsEval=__import__('copy').deepcopy(E.params)
    with contextlib.redirect_stdout(io.StringIO()): E.accumulate(); E.summarize()
    return E.stats[0], E.stats[3:6]
def per_image_ap(cache, area=0):
    """per-image AP (mean over categories with non-ignored GT, 10 IoU thr, 101 recall pts, maxDet 100); nan if no GT."""
    I=len(cache['imgIds']); K=len(cache['catIds']); A=4; ev=cache['evalImgs']; rec=np.linspace(0,1,101)
    s=np.zeros(I); c=np.zeros(I)
    for k in range(K):
        for i in range(I):
            e=ev[k*A*I+area*I+i]
            if e is None: continue
            npig=int(np.sum(np.asarray(e['gtIgnore'])==0))
            if npig==0: continue
            sc=np.asarray(e['dtScores']); o=np.argsort(-sc,kind='mergesort')
            dm=np.asarray(e['dtMatches'])[:,o]; di=np.asarray(e['dtIgnore'])[:,o]
            if dm.shape[1]==0: c[i]+=1; continue
            tp=np.cumsum((dm!=0)&~di,1).astype(float); fp=np.cumsum((dm==0)&~di,1).astype(float)
            ap=0.
            for t in range(tp.shape[0]):
                r=tp[t]/npig; p=tp[t]/np.maximum(tp[t]+fp[t],1e-12)
                p=np.maximum.accumulate(p[::-1])[::-1]
                ids=np.searchsorted(r,rec,side='left'); q=np.where(ids<len(p),p[np.minimum(ids,len(p)-1)],0.)
                ap+=q.mean()
            s[i]+=ap/tp.shape[0]; c[i]+=1
    out=np.full(I,np.nan); m=c>0; out[m]=s[m]/c[m]; return out
