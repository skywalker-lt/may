# K-source exact per-image mixing and subset AP from cached COCOeval evalImgs (prep_eval.py pickles; area=all, maxDet 100).
# AP(sel, S): detections of source sel[i] on image i, images restricted to S (bool mask), GT positives counted on S only.
# Optional image weights w (bootstrap multiplicities). Categories with no positives in S are skipped (pycocotools -1 rule).
import pickle, numpy as np
class MixK:
    def __init__(self, paths):
        Ds=[pickle.load(open(p,'rb')) for p in paths]
        D0=Ds[0]
        for D in Ds[1:]: assert D['imgIds']==D0['imgIds'] and D['catIds']==D0['catIds']
        self.imgIds=D0['imgIds']; self.catIds=D0['catIds']; self.K=len(Ds)
        nI=len(self.imgIds); nA=len(D0['areaRng']); nK=len(self.catIds)
        self.img_index={i:j for j,i in enumerate(self.imgIds)}
        self.recThrs=D0['recThrs']; self.T=len(D0['iouThrs'])
        self.npig=np.zeros((nI,nK))
        self.blk=[]
        for k in range(nK):
            sc=[];im=[];dm=[];di=[];sr=[]
            for s,D in enumerate(Ds):
                for i in range(nI):
                    e=D['recs'][k*nA*nI+i]
                    if e is None: continue
                    g=int(np.count_nonzero(np.asarray(e[5])==0))
                    if s==0 or self.npig[i,k]==0: self.npig[i,k]=max(self.npig[i,k],g)
                    n=min(len(e[2]),100)
                    if n==0: continue
                    sc.append(e[2][:100]); im.append(np.full(n,i)); dm.append(np.asarray(e[3])[:,:100]); di.append(np.asarray(e[4])[:,:100]); sr.append(np.full(n,s))
            sc=np.concatenate(sc); o=np.argsort(-sc,kind='mergesort')
            self.blk.append((np.concatenate(im)[o].astype(np.int64), np.concatenate(dm,1)[:,o].astype(bool), np.concatenate(di,1)[:,o].astype(bool), np.concatenate(sr)[o].astype(np.int8)))
    def ap(self, sel, S=None, w=None, per_cat=False):
        nI=len(self.imgIds); sel=np.asarray(sel,np.int8)
        if S is None: S=np.ones(nI,bool)
        wi=S.astype(float) if w is None else S*w
        R=len(self.recThrs); out=[]
        for k,(im,dm,di,sr) in enumerate(self.blk):
            npig=float(self.npig[:,k]@wi)
            if npig<=0: out.append(np.nan); continue
            keep=(sr==sel[im])&(wi[im]>0)
            ww=wi[im[keep]]; m=dm[:,keep]; ig=di[:,keep]
            tp=np.cumsum((m&~ig)*ww,1); fp=np.cumsum((~m&~ig)*ww,1)
            acc=0.0
            for t in range(self.T):
                rc=tp[t]/npig; pr=tp[t]/(tp[t]+fp[t]+np.spacing(1))
                pr=np.maximum.accumulate(pr[::-1])[::-1]
                idx=np.searchsorted(rc,self.recThrs,side='left'); ok=idx<len(pr)
                q=np.zeros(R); q[ok]=pr[idx[ok]]; acc+=q.mean()
            out.append(acc/self.T)
        out=np.array(out)
        return out if per_cat else np.nanmean(out)
