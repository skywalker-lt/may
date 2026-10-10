import sys, json, numpy as np, io, contextlib
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent9')
from fastap import ap_from, ap_mix, ap_rec
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.stats import spearmanr
R='/data/tmp/ds-yolo/seminar5/work/agent9/r2'
with contextlib.redirect_stdout(io.StringIO()): gt=COCO('/data/tmp/ds-yolo/seminar5/inputs/dumps/instances_val2017.json')
sub=json.load(open(f'{R}/fq_ids.json')); st={d['image_id']:d for d in json.load(open(f'{R}/fq_stats.json'))}
catIds=sorted(gt.getCatIds()); ci={c:k for k,c in enumerate(catIds)}; ii={im:i for i,im in enumerate(sub)}; n=len(sub)
def rec_of(dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(dets); E=COCOeval(gt,dt,'bbox'); E.params.imgIds=sub; E.params.areaRng=[[0,1e10]]; E.params.areaRngLbl=['all']; E.params.maxDets=[100]; E.evaluate()
    cats=[];imgs=[];sc=[];m=[];ig=[];npig=np.zeros((80,n),np.int32)
    for e in E.evalImgs:
        if e is None: continue
        k=ci[e['category_id']]; i=ii[e['image_id']]; npig[k,i]=int(np.sum(np.logical_not(e['gtIgnore'])))
        if len(e['dtScores'])==0: continue
        cats.append(np.full(len(e['dtScores']),k)); imgs.append(np.full(len(e['dtScores']),i)); sc.append(np.array(e['dtScores'])); m.append(e['dtMatches']>0); ig.append(e['dtIgnore'].astype(bool))
    return dict(cat=np.concatenate(cats),img=np.concatenate(imgs),score=np.concatenate(sc),match=np.concatenate(m,1),ign=np.concatenate(ig,1),npig=npig)
def full_ap(dets):
    with contextlib.redirect_stdout(io.StringIO()):
        dt=gt.loadRes(dets); E=COCOeval(gt,dt,'bbox'); E.params.imgIds=sub; E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0]
recs={k:rec_of(json.load(open(f'{R}/fq_{k}.json'))) for k in ('fp','max','p9999')}
pub=[d for d in json.load(open('/data/tmp/ds-yolo/seminar5/inputs/dumps/dumpml_yolo26m_coco.json')) if d['image_id'] in ii]
print('subset of %d val2017 images: official-protocol AP of public dump %.4f; CPU fp32 %.4f; fake-INT8 tail max-calib %.4f; p99.99-calib %.4f'%(
    n,full_ap(pub),full_ap(json.load(open(f'{R}/fq_fp.json'))),full_ap(json.load(open(f'{R}/fq_max.json'))),full_ap(json.load(open(f'{R}/fq_p9999.json')))),flush=True)
cnt=recs['fp']['npig'].sum(0)
def perimg(r):
    o=np.full(n,np.nan)
    for i in range(n):
        if cnt[i]==0: continue
        s=r['img']==i; o[i]=ap_from(r['cat'][s],r['score'][s],r['match'][:,s],r['ign'][:,s],r['npig'][:,i])
    return o
pf=perimg(recs['fp'])
feat={k:np.array([st[im][k] for im in sub]) for k in ('l5_ratio','l5_frac','clip_mean','clip_max')}; feat['GT count']=cnt.astype(float)
rng=np.random.default_rng(1)
for q in ('max','p9999'):
    pq=perimg(recs[q]); loss=pf-pq; ok=~np.isnan(loss)
    A_fp=ap_rec(recs['fp']); A_q=ap_rec(recs[q]); d=A_fp-A_q
    print(f'[{q}] delta (fp - int8 tail, match-cache AP) = {d:+.4f}; images with |per-image change|>0.01: {np.mean(np.abs(loss[ok])>0.01):.2f}',flush=True)
    print('   Spearman(per-image INT8 loss, feature): '+', '.join(f'{k} {spearmanr(loss[ok],v[ok])[0]:+.3f}' for k,v in feat.items()),flush=True)
    for s in (0.2,0.3):
        k=int(s*n)
        def mix(score):
            A=np.zeros(n,int); A[np.argsort(-score)[:k]]=1; return ap_mix([recs[q],recs['fp']],A)
        nul=np.array([ap_mix([recs[q],recs['fp']],rng.permutation(np.r_[np.ones(k,int),np.zeros(n-k,int)])) for _ in range(20)])
        line=f'   fp share {s}: null {nul.mean():.4f} (sd {nul.std():.4f}); oracle {mix(np.nan_to_num(loss,nan=-9)):.4f}'
        for kk,v in feat.items():
            a=mix(v); g=a-nul.mean(); line+=f'; {kk} {a:.4f} (g {g:+.4f}, r {g/((1-s)*d) if d>0 else float("nan"):.2f})'
        print(line,flush=True)
