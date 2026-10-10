# One pycocotools evaluate() on a dump; pickle evalImgs-derived arrays so that any
# per-(image,class) monotone rescoring can be re-accumulated without re-matching.
import sys, json, pickle, numpy as np, time, io, contextlib
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
gt_path = '/data/tmp/ds-yolo/seminar5/inputs/dumps/instances_val2017.json'
dump, out = sys.argv[1], sys.argv[2]
t=time.time()
with contextlib.redirect_stdout(io.StringIO()):
    gt = COCO(gt_path)
    dt = gt.loadRes(dump)
E = COCOeval(gt, dt, 'bbox')
with contextlib.redirect_stdout(io.StringIO()):
    E.evaluate()
E.accumulate(); E.summarize()
print('base AP', E.stats[0], 'time', time.time()-t)
# store evalImgs compactly: for each entry keep imgId, catId, aRng idx, maxDet, dtScores, dtMatches, dtIgnore, gtIgnore
p = E.params
recs=[]
for e in E.evalImgs:
    if e is None: recs.append(None); continue
    recs.append((e['image_id'], e['category_id'], np.array(e['dtScores'],np.float64), e['dtMatches'], e['dtIgnore'], e['gtIgnore']))
pickle.dump(dict(recs=recs, catIds=p.catIds, areaRng=p.areaRng, imgIds=p.imgIds, iouThrs=p.iouThrs, recThrs=p.recThrs, maxDets=p.maxDets, base=E.stats.tolist()), open(out,'wb'))
