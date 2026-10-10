# Run pycocotools evaluate() once per dump (area 'all', maxDet 100) and store per-detection match records,
# so that any per-image mixture or per-(image,class) rescoring can be re-accumulated without re-matching.
import os, sys, json, time, pickle, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D=sys.argv[1]; GTD='/data/tmp/ds-yolo/seminar5/inputs/dumps'
OUT='/data/tmp/ds-yolo/seminar5/work/agent9/cache'
gt=COCO(f'{GTD}/instances_val2017.json')
imgIds=sorted(gt.getImgIds()); catIds=sorted(gt.getCatIds())
img_index={im:i for i,im in enumerate(imgIds)}; cat_index={c:k for k,c in enumerate(catIds)}
for name in sys.argv[2:]:
    t=time.time()
    dt=gt.loadRes(f'{D}/{name}.json')
    E=COCOeval(gt,dt,'bbox'); E.params.imgIds=imgIds; E.params.areaRng=[[0,1e10]]; E.params.areaRngLbl=['all']; E.params.maxDets=[100]
    E.evaluate()
    cats=[];imgs=[];scores=[];match=[];ign=[]; npig=np.zeros((len(catIds),len(imgIds)),np.int32)
    for e in E.evalImgs:
        if e is None: continue
        k=cat_index[e['category_id']]; i=img_index[e['image_id']]
        npig[k,i]=int(np.sum(np.logical_not(e['gtIgnore'])))
        n=len(e['dtScores'])
        if n==0: continue
        cats.append(np.full(n,k,np.int16)); imgs.append(np.full(n,i,np.int32)); scores.append(np.array(e['dtScores'],np.float64))
        match.append(e['dtMatches']>0); ign.append(e['dtIgnore'].astype(bool))
    rec=dict(cat=np.concatenate(cats),img=np.concatenate(imgs),score=np.concatenate(scores),
             match=np.concatenate(match,1),ign=np.concatenate(ign,1),npig=npig,imgIds=imgIds,catIds=catIds)
    pickle.dump(rec,open(f'{OUT}/{name}.pkl','wb'),protocol=4)
    print(name,'dets',len(rec['score']),'time %.0fs'%(time.time()-t),flush=True)
