import sys; sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent9')
import numpy as np, json
from common import *
M=load('dumpml_yolo26m_coco'); V={t:load(f'dumpml_yolo26m_nms{t}') for t in (70,60,50)}
n=len(M['imgIds']); Y=gt_matrix(M); cnt=Y.sum(1); F=feats(M)
print('M %.4f'%ap_rec(M), ' '.join('nms%d %.4f'%(t,ap_rec(v)) for t,v in V.items()),flush=True)
def perimg(r):
    out=np.full(n,np.nan)
    order=np.argsort(r['img'],kind='stable'); bnd=np.searchsorted(r['img'][order],np.arange(n+1))
    for i in range(n):
        if cnt[i]==0: continue
        sel=order[bnd[i]:bnd[i+1]]
        out[i]=ap_from(r['cat'][sel],r['score'][sel],r['match'][:,sel],r['ign'][:,sel],r['npig'][:,i])
    return out
pM=perimg(M)
# GT-free router features from M's own output: count of dets >= 0.25; same-class confident pairs IoU>0.5
dets=json.load(open('/data/tmp/ds-yolo/seminar5/inputs/dumps/dumpml_yolo26m_coco.json'))
ii={im:i for i,im in enumerate(M['imgIds'])}
own=np.zeros(n); pairs=np.zeros(n); import collections; G=collections.defaultdict(list)
for d in dets:
    if d['score']>=0.25: G[(ii[d['image_id']],d['category_id'])].append(d['bbox'])
for (i,c),bb in G.items():
    own[i]+=len(bb); b=np.array(bb); x1,y1=b[:,0],b[:,1]; x2,y2=x1+b[:,2],y1+b[:,3]; ar=b[:,2]*b[:,3]
    for j in range(len(b)):
        xx1=np.maximum(x1[j],x1[j+1:]); yy1=np.maximum(y1[j],y1[j+1:]); xx2=np.minimum(x2[j],x2[j+1:]); yy2=np.minimum(y2[j],y2[j+1:])
        inter=np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None); pairs[i]+=(inter/(ar[j]+ar[j+1:]-inter+1e-9)>0.5).sum()
from scipy.stats import spearmanr
rng=np.random.default_rng(7)
for t,v in V.items():
    pv=perimg(v); g=pv-pM; ok=~np.isnan(g)
    print(f'nms{t}: images changed {np.sum(np.abs(g[ok])>1e-9)}, mean per-image gain {np.nanmean(g):+.4f}; Spearman gain vs own count {spearmanr(g[ok],own[ok])[0]:+.3f}, vs conf pairs {spearmanr(g[ok],pairs[ok])[0]:+.3f}, vs GT count {spearmanr(g[ok],cnt[ok])[0]:+.3f}',flush=True)
    for share in (0.1,0.2,0.3):
        k=int(share*n)
        def mix(score):
            A=np.zeros(n,int); A[np.argsort(-score)[:k]]=1; return ap_mix([M,v],A)
        orc=mix(np.nan_to_num(g,nan=-1)); r1=mix(pairs+1e-3*own); r2=mix(own); 
        nul=np.mean([ap_mix([M,v],rng.permutation(np.r_[np.ones(k,int),np.zeros(n-k,int)])) for _ in range(3)])
        print(f'  share {share}: oracle {orc:.4f}  null {nul:.4f}  own conf-pair router {r1:.4f} ({r1-nul:+.4f})  own count router {r2:.4f} ({r2-nul:+.4f})',flush=True)
