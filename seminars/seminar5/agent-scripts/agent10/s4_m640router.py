# Same simulated mixture (delta 0.013, share 0.3) routed by a count head on M's own 640 stem features (free inside SRP).
import sys, numpy as np, pickle
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent10')
from common import *
from s2_features import oof_ridge
from scipy.stats import spearmanr
F=pickle.load(open(W+'/features.pkl','rb')); ids=F['ids']
z=np.load(D+'/val2017_stem_pooled.npz'); assert (z['image_id']==ids).all()
ps=oof_ridge(z['m640'].astype(np.float64),np.log1p(F['small']))
print('spearman m640-pred vs GT small %.3f'%spearmanr(ps,F['small'])[0])
P=pickle.load(open(W+'/pert_0.06_0.15.pkl','rb'))
bC=by_image(load('dumpml_yolo26m_coco.json')); bN=by_image(P['dets'])
k=int(round(0.3*len(ids))); r=np.zeros(len(ids),bool); r[np.argsort(-ps,kind='mergesort')[:k]]=True
res=evaluate(mix(bN,bC,r,ids)); print('share 0.30 learned m640 small count AP=%.4f'%res['AP'])
