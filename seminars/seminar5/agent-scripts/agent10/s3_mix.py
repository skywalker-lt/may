# Precision-mixture headroom under the simulated INT8 loss: clean (fp16) vs perturbed (INT8 proxy) per image.
import sys, numpy as np, pickle
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent10')
from common import *
from scipy.stats import spearmanr
tag=sys.argv[1]; shares=[] if sys.argv[2]=='none' else [float(s) for s in sys.argv[2].split(',')]
F=pickle.load(open(W+'/features.pkl','rb')); ids=F['ids']
clean=pickle.load(open(W+'/pi_m640.pkl','rb')); P=pickle.load(open(W+f'/pert_{tag}.pkl','rb'))
pc=clean['pi']; pn=P['res']['pi']
d=np.array([pc.get(int(i),np.nan)-pn.get(int(i),np.nan) for i in ids]); has=~np.isnan(d)
print('AP clean %.4f noisy %.4f delta %.4f'%(clean['AP'],P['res']['AP'],clean['AP']-P['res']['AP']))
print('per-image dAP (images with GT %d): mean %.4f sd %.4f, share of images with dAP>0: %.2f, top-30%% of images carry %.2f of the summed dAP'%(
    has.sum(),np.nanmean(d),np.nanstd(d),np.mean(d[has]>0),np.sort(d[has])[::-1][:int(0.3*has.sum())].sum()/d[has].sum()))
for k in ['cnt','small','pred_cnt','pred_small']:
    print('  spearman dAP vs %-10s %.3f'%(k,spearmanr(d[has],F[k][has])[0]))
# noise-only split-half: is dAP itself reproducible? (second noise seed would be needed; skip)
bC=by_image(load('dumpml_yolo26m_coco.json')); bN=by_image(P['dets'])
rng=np.random.default_rng(7); dz=np.where(has,d,0.0)
for s in shares:
    k=int(round(s*len(ids)))
    def top(v): r=np.zeros(len(ids),bool); r[np.argsort(-v,kind='mergesort')[:k]]=True; return r
    routes={'learned n320 small count':top(F['pred_small']),'GT small count':top(F['small']),'oracle dAP':top(dz)}
    for j in range(2):
        r=np.zeros(len(ids),bool); r[rng.permutation(len(ids))[:k]]=True; routes[f'null {j}']=r
    for name,r in routes.items():
        res=evaluate(mix(bN,bC,r,ids))
        print('  share %.2f %-26s AP=%.4f S/M/L=%.4f/%.4f/%.4f'%(s,name,res['AP'],res['APS'],res['APM'],res['APL']),flush=True)
