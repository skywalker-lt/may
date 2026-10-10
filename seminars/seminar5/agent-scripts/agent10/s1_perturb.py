# Simulated precision loss: stride-scaled box-side noise + logit noise on m640 dumps; per-image AP for clean and noisy.
import sys, numpy as np, pickle, time
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent10')
from common import *
G=gt()
imgs={i['id']:i for i in G.dataset['images']}
def perturb(dets, ss, sl, seed):
    rng=np.random.default_rng(seed); out=[]
    for x in dets:
        im=imgs[x['image_id']]; r=640/max(im['width'],im['height'])
        x0,y0,w,h=x['bbox']; sz=np.sqrt(max(w*h,1e-6))*r
        stride=8 if sz<64 else (16 if sz<160 else 32)
        n=rng.normal(0,ss*stride/r,4)
        x1,y1,x2,y2=x0+n[0],y0+n[1],x0+w+n[2],y0+h+n[3]
        if x2<=x1: x2=x1+1e-3
        if y2<=y1: y2=y1+1e-3
        s=min(max(x['score'],1e-6),1-1e-6); lg=np.log(s/(1-s))+rng.normal(0,sl)
        out.append({'image_id':x['image_id'],'category_id':x['category_id'],'bbox':[x1,y1,x2-x1,y2-y1],'score':float(1/(1+np.exp(-lg)))})
    return out
base=load('dumpml_yolo26m_coco.json')
import os
if not os.path.exists(W+'/pi_m640.pkl'):
    r=evaluate(base,True); pickle.dump(r,open(W+'/pi_m640.pkl','wb')); print('clean',r['AP'],flush=True)
for ss,sl in [(float(a),float(b)) for a,b in (z.split(',') for z in sys.argv[1:])]:
    p=perturb(base,ss,sl,1); r2=evaluate(p,True)
    print('noise ss=%.2f sl=%.2f'%(ss,sl),{k:round(v,4) for k,v in r2.items() if k!='pi'},flush=True)
    pickle.dump({'dets':p,'res':r2},open(W+f'/pert_{ss}_{sl}.pkl','wb'))
