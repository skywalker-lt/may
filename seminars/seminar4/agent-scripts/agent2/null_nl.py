import json, os, numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D="/data/tmp/ds-yolo/seminar4/inputs/dumps"
gt=COCO(os.path.join(D,"instances_val2017.json")); IMG=sorted(gt.getImgIds())
n=json.load(open(os.path.join(D,"dump_yolo26n_coco.json"))); l=json.load(open(os.path.join(D,"dump_yolo26l_coco.json")))
rng=np.random.default_rng(0); pick=set(np.array(IMG)[rng.random(len(IMG))<0.538].tolist())
res=[d for d in n if d["image_id"] not in pick]+[d for d in l if d["image_id"] in pick]
ev=COCOeval(gt,gt.loadRes(res),"bbox"); ev.params.imgIds=IMG; ev.evaluate(); ev.accumulate(); ev.summarize()
print(f"RESULT random n/l p_l=0.538 (seed 0): AP={ev.stats[0]:.4f}")
