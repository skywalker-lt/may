# COCO-style AP (101-point interpolated, 10 IoU thresholds, area all, maxDet 100) from cached match records.
import numpy as np
def ap_from(cat,score,match,ign,npig_k,ncat=80):
    """cat: (n,), score: (n,), match/ign: (10,n) bool, npig_k: (ncat,) gt counts. Returns mean AP over cats with gt."""
    rs=np.linspace(0,1,101); aps=[]
    order=np.lexsort((-score,cat))  # by cat then score desc (stable mergesort in pycocotools; ties negligible)
    cat_s=cat[order]; bounds=np.searchsorted(cat_s,np.arange(ncat+1))
    for k in range(ncat):
        G=npig_k[k]
        if G==0: continue
        a,b=bounds[k],bounds[k+1]
        idx=order[a:b]
        m=match[:,idx]; ig=ign[:,idx]
        tp=np.cumsum(m & ~ig,1).astype(float); fp=np.cumsum(~m & ~ig,1).astype(float)
        apk=[]
        for t in range(10):
            if tp.shape[1]==0: apk.append(0.0); continue
            rc=tp[t]/G; pr=tp[t]/np.maximum(tp[t]+fp[t],np.spacing(1))
            pr=np.maximum.accumulate(pr[::-1])[::-1]
            inds=np.searchsorted(rc,rs,side='left')
            q=np.zeros(101); ok=inds<len(pr); q[ok]=pr[inds[ok]]
            apk.append(q.mean())
        aps.append(np.mean(apk))
    return float(np.mean(aps))
def ap_rec(rec,imgmask=None,score=None):
    s=rec['score'] if score is None else score
    if imgmask is None:
        return ap_from(rec['cat'],s,rec['match'],rec['ign'],rec['npig'].sum(1))
    sel=imgmask[rec['img']]
    return ap_from(rec['cat'][sel],s[sel],rec['match'][:,sel],rec['ign'][:,sel],rec['npig'][:,imgmask].sum(1))
def ap_mix(recs,assign):
    """recs: list of records; assign: (nimg,) index of record used for each image."""
    cats=[];sc=[];m=[];ig=[];np_=0
    for j,r in enumerate(recs):
        mask=(assign==j); sel=mask[r['img']]
        cats.append(r['cat'][sel]); sc.append(r['score'][sel]); m.append(r['match'][:,sel]); ig.append(r['ign'][:,sel])
        np_=np_+r['npig'][:,mask].sum(1)
    return ap_from(np.concatenate(cats),np.concatenate(sc),np.concatenate(m,1),np.concatenate(ig,1),np_)
