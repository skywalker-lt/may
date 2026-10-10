import json, numpy as np, os, io, contextlib, pickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D='/data/tmp/ds-yolo/seminar5/inputs/dumps'
W='/data/tmp/ds-yolo/seminar5/work/agent10'
_gt=None
def gt():
    global _gt
    if _gt is None:
        with contextlib.redirect_stdout(io.StringIO()):
            _gt=COCO(D+'/instances_val2017.json')
    return _gt
def load(name):
    return json.load(open(f'{D}/{name}'))
def evaluate(dets, per_image=False):
    G=gt()
    with contextlib.redirect_stdout(io.StringIO()):
        dt=G.loadRes(dets) if len(dets) else COCO()
        E=COCOeval(G,dt,'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    s=E.stats
    out={'AP':s[0],'AP50':s[1],'AP75':s[2],'APS':s[3],'APM':s[4],'APL':s[5]}
    if per_image: out['pi']=per_image_ap(E)
    return out
def per_image_ap(E):
    """per-image AP: mean over (category with non-ignored GT in image) of 101-pt AP over 10 IoU thr, area all, maxDet 100"""
    p=E.params; nA=len(p.areaRng); nI=len(p.imgIds)
    rs=np.linspace(0,1,101)
    acc={}
    for k,cat in enumerate(p.catIds):
        for i,img in enumerate(p.imgIds):
            e=E.evalImgs[k*nA*nI+0*nI+i]
            if e is None: continue
            gi=np.array(e['gtIgnore'],bool); npos=(~gi).sum()
            if npos==0: continue
            sc=np.array(e['dtScores']); o=np.argsort(-sc,kind='mergesort')
            dm=e['dtMatches'][:,o]; dig=e['dtIgnore'][:,o]
            aps=[]
            for t in range(dm.shape[0]):
                keep=~dig[t]
                tp=(dm[t][keep]>0).astype(float); fp=1-tp
                if len(tp)==0: aps.append(0.0); continue
                ctp=np.cumsum(tp); cfp=np.cumsum(fp)
                rc=ctp/npos; pr=ctp/(ctp+cfp)
                pr=np.maximum.accumulate(pr[::-1])[::-1]
                idx=np.searchsorted(rc,rs,side='left')
                q=np.where(idx<len(pr),pr[np.minimum(idx,len(pr)-1)],0.0)
                aps.append(q.mean())
            acc.setdefault(img,[]).append(np.mean(aps))
    return {img:float(np.mean(v)) for img,v in acc.items()}
def by_image(dets):
    d={}
    for x in dets: d.setdefault(x['image_id'],[]).append(x)
    return d
def mix(bA,bB,routeB,img_ids):
    out=[]
    for i,img in enumerate(img_ids):
        out+= (bB if routeB[i] else bA).get(img,[])
    return out
