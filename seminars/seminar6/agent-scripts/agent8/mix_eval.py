"""Build a per-image mixture of protocol dumps and score it with pycocotools (multi-label dumps, full val2017).
usage: mix_eval.py <label> <seed> name=path:share [name=path:share ...] [miss=<fraction>]
Images are assigned to rungs at random at the given shares (a share-matched random route = budget route whose
signal is independent of the image); 'miss' images get no detections (a dropped frame).
"""
import sys, json, numpy as np, time, io, contextlib
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ANN='/data/tmp/ds-yolo/seminar6/inputs/dumps/instances_val2017.json'
label=sys.argv[1]; seed=int(sys.argv[2]); rungs=[]; miss=0.0
for a in sys.argv[3:]:
    if a.startswith('miss='): miss=float(a[5:]); continue
    nm,rest=a.split('='); path,share=rest.rsplit(':',1); rungs.append((nm,path,float(share)))
with contextlib.redirect_stdout(io.StringIO()):
    gt=COCO(ANN)
ids=sorted(gt.getImgIds()); rng=np.random.default_rng(seed)
shares=np.array([s for _,_,s in rungs]); shares=shares/shares.sum()*(1-miss)
cut=np.cumsum(np.r_[shares,miss]); u=rng.random(len(ids)); assign=np.searchsorted(cut,u)
out=[]; counts={}
for k,(nm,path,_) in enumerate(rungs):
    sel=set(int(i) for i,a in zip(ids,assign) if a==k); counts[nm]=len(sel)
    d=json.load(open(path)); out+= [x for x in d if x['image_id'] in sel]
counts['miss']=int(np.sum(assign==len(rungs)))
with contextlib.redirect_stdout(io.StringIO()):
    dt=gt.loadRes(out); E=COCOeval(gt,dt,'bbox'); E.evaluate(); E.accumulate(); E.summarize()
print(f'{label} seed={seed} shares={ {k:round(v/len(ids),3) for k,v in counts.items()} } AP={E.stats[0]:.4f} AP50={E.stats[1]:.4f} AP75={E.stats[2]:.4f} APs/m/l={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f}')
