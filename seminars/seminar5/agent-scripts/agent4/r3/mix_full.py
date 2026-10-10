# One thread. Full-val2017 exact mixtures of public dumps under the train-fitted SA-2 router (route list of if_shrink_p4attn0).
import os, json, sys, numpy as np; os.environ['OMP_NUM_THREADS']='1'
from pycocotools.coco import COCO; from pycocotools.cocoeval import COCOeval
import contextlib, io
D='/data/tmp/ds-yolo/seminar5/inputs/'
gt=COCO(D+'dumps/instances_val2017.json')
def load(p):
    d=json.load(open(p)); by={}
    for x in d: by.setdefault(x['image_id'],[]).append(x)
    return by
M={'m512':D+'dumps/dumpml_yolo26m_512_coco.json','m640':D+'dumps/dumpml_yolo26m_coco.json',
   'l512':D+'dumps_r2/dumpml_yolo26l_512_coco.json','l640':D+'dumps/dump_yolo26l_coco.json','l448':D+'dumps_r2/dumpml_yolo26l_448_coco.json'}
B={k:load(v) for k,v in M.items()}
ids=sorted(gt.getImgIds())
route=json.load(open('/data/tmp/ds-yolo/seminar5/work/agent4/request/if_shrink_p4attn0.routes_val2017.json'))
score=np.array([route[str(i)][0] for i in ids]); eng=np.array([route[str(i)][1] for i in ids])
def ev(sel):  # sel: dict image->model key
    dets=[x for i in ids for x in B[sel[i]].get(i,[])]
    with contextlib.redirect_stdout(io.StringIO()):
        E=COCOeval(gt,gt.loadRes(dets),'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0]
def mix(lo,hi,mask): return {i:(lo if m else hi) for i,m in zip(ids,mask)}
rng=np.random.default_rng(0)
print('engine route share',eng.mean())
for lo,hi in (('m512','m640'),('l512','l640'),('l448','l640')):
    print(lo,hi,'engine route',round(ev(mix(lo,hi,eng)),4),flush=True)
    for q in (0.4,0.6):
        thr=np.quantile(score,q); m=score<thr
        print(lo,hi,'share',q,round(ev(mix(lo,hi,m)),4),flush=True)
    for r in range(2):
        m=np.zeros(len(ids),bool); m[rng.choice(len(ids),int(eng.sum()),replace=False)]=True
        print(lo,hi,'null draw',r,round(ev(mix(lo,hi,m)),4),flush=True)
