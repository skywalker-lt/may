# Gate K0' reader. One thread. Works on the CPU subset (sub_*.json, every 5th val id) or on full-val pod dumps (args).
# Reliance R(h) = AP(public) - AP(no-attention) on route half h; reliance-routed mixture = public on the shrink half,
# no-attention elsewhere, against the same with a random half (5 draws).
import os, sys, json, numpy as np, contextlib, io; os.environ['OMP_NUM_THREADS']='1'
from pycocotools.coco import COCO; from pycocotools.cocoeval import COCOeval
gt=COCO('/data/tmp/ds-yolo/seminar5/inputs/dumps/instances_val2017.json')
route=json.load(open('/data/tmp/ds-yolo/seminar5/work/agent4/request/if_shrink_p4attn0.routes_val2017.json'))
def load(p):
    by={}
    for x in json.load(open(p)): by.setdefault(x['image_id'],[]).append(x)
    return by
pairs=[tuple(a.split(',')) for a in sys.argv[1:]]   # label,public_json,noattn_json
for lab,pp,pn in pairs:
    P,N=load(pp),load(pn)
    ids=sorted(set(gt.getImgIds()) & (set(P)|set(N))) if 'sub_' in pp else sorted(gt.getImgIds())
    shr=np.array([route[str(i)][1] for i in ids])
    def ev(src_by_img, sub):
        dets=[x for i in sub for x in src_by_img(i)]
        with contextlib.redirect_stdout(io.StringIO()):
            E=COCOeval(gt,gt.loadRes(dets),'bbox'); E.params.imgIds=list(sub); E.evaluate(); E.accumulate(); E.summarize()
        return E.stats[0]
    res={}
    for hname,sub in (('all',ids),('shrink',[i for i,s in zip(ids,shr) if s]),('keep',[i for i,s in zip(ids,shr) if not s])):
        a=ev(lambda i:P.get(i,[]),sub); b=ev(lambda i:N.get(i,[]),sub); res[hname]=(a,b)
        print(f'{lab} {hname:6s} n={len(sub)} public {a:.4f} noattn {b:.4f} reliance {a-b:+.4f}',flush=True)
    rs,rk=res['shrink'][0]-res['shrink'][1],res['keep'][0]-res['keep'][1]
    print(f'{lab} reliance ratio shrink/keep = {rs/rk if rk!=0 else float("nan"):.2f}  (K0 gate >= 1.3)')
    sset=set(i for i,s in zip(ids,shr) if s)
    m=ev(lambda i:P.get(i,[]) if i in sset else N.get(i,[]),ids)
    rng=np.random.default_rng(0); nulls=[]
    for r in range(5):
        rs_=set(rng.choice(ids,len(sset),replace=False).tolist())
        nulls.append(ev(lambda i:P.get(i,[]) if i in rs_ else N.get(i,[]),ids))
    print(f'{lab} routed mixture {m:.4f} null mean {np.mean(nulls):.4f} (sd {np.std(nulls):.4f}) lift {m-np.mean(nulls):+.4f} (K0 gate >= +0.0015)',flush=True)
