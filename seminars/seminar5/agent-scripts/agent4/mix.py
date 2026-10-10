# Mix per-image detections of two dumps under a route; score with pycocotools (full val2017).
import numpy as np, json, contextlib, io, sys, time
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
W='/data/tmp/ds-yolo/seminar5/work/agent4/'; D='/data/tmp/ds-yolo/seminar5/inputs/dumps/'
with contextlib.redirect_stdout(io.StringIO()): gt=COCO(D+'instances_val2017.json')
A=json.load(open(D+'dump_yolo11m_coco.json')); B=json.load(open(D+'dump_yolov12m_sdpa_coco.json'))  # B = attention family
f=np.load(W+'feat.npz'); ids=f['ids']; pa=np.load(W+'pi_y11m.npz')['ap']; pb=np.load(W+'pi_y12m.npz')['ap']
d=np.nan_to_num(pb-pa); predc=np.load(W+'pred_logcount_m640.npy')
z=np.load(D+'val2017_stem_pooled.npz'); zi={int(i):j for j,i in enumerate(z['image_id'])}; m640=z['m640'][[zi[int(i)] for i in ids]]
def ridge_oof(X,y,lam=100.,k=5,seed=0):
    rng=np.random.default_rng(seed); fo=rng.integers(0,k,len(y)); p=np.zeros(len(y))
    X=(X-X.mean(0))/(X.std(0)+1e-6); X=np.c_[X,np.ones(len(X))]
    for j in range(k):
        tr=fo!=j; p[fo==j]=X[fo==j]@np.linalg.solve(X[tr].T@X[tr]+lam*np.eye(X.shape[1]),X[tr].T@y[tr])
    return p
learned=ridge_oof(m640,d)
def top(score,s): 
    thr=np.quantile(score,1-s); return set(ids[score>thr].tolist())
def ev(name,selB):
    t=time.time(); dets=[x for x in A if x['image_id'] not in selB]+[x for x in B if x['image_id'] in selB]
    with contextlib.redirect_stdout(io.StringIO()):
        E=COCOeval(gt,gt.loadRes(dets),'bbox'); E.evaluate(); E.accumulate(); E.summarize()
    print(f'{name:55s} share={len(selB)/len(ids):.3f} AP={E.stats[0]:.4f} S/M/L={E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f} ({time.time()-t:.0f}s)',flush=True)
rng=np.random.default_rng(1)
s=float(sys.argv[1]) if len(sys.argv)>1 else 0.5
ev(f'oracle (per-image proxy, best {s})',top(d+1e-9*rng.random(len(d)),s))
ev(f'null random {s} draw 1',top(rng.random(len(d)),s))
ev(f'null random {s} draw 2',top(rng.random(len(d)),s))
ev(f'GT count rule {s}',top(f['count']+1e-3*rng.random(len(d)),s))
ev(f'learned m640 count {s}',top(predc,s))
ev(f'learned m640 ridge on gain {s}',top(learned,s))
ev(f'GT large-object count rule {s}',top(f['large']+1e-3*rng.random(len(d)),s))
