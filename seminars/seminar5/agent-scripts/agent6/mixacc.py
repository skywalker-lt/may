# Exact per-image mixing of two dumps via cached COCOeval evalImgs (area=all, maxDet 100), plus per-image marginal gains.
import pickle, numpy as np
class Mix:
    def __init__(self, pA, pB):
        DA = pickle.load(open(pA,'rb')); DB = pickle.load(open(pB,'rb'))
        assert DA['imgIds']==DB['imgIds'] and DA['catIds']==DB['catIds']
        self.imgIds = DA['imgIds']; self.catIds = DA['catIds']
        nI=len(self.imgIds); nA=len(DA['areaRng']); nK=len(self.catIds)
        self.img_index={i:j for j,i in enumerate(self.imgIds)}
        self.recThrs=DA['recThrs']; self.T=len(DA['iouThrs'])
        self.blk=[]; self.touch=[[] for _ in range(nI)]
        for k in range(nK):
            parts=[]; npig=None
            for src,D in enumerate((DA,DB)):
                Es=[D['recs'][k*nA*nI + i] for i in range(nI)]  # area 0
                sc=[];im=[];dm=[];di=[]; g=0
                for e in Es:
                    if e is None: continue
                    n=min(len(e[2]),100); ii=self.img_index[e[0]]
                    sc.append(e[2][:100]); im.append(np.full(n,ii)); dm.append(e[3][:,:100]); di.append(e[4][:,:100])
                    g+=int(np.count_nonzero(np.asarray(e[5])==0))
                    if src==0 or True: self.touch[ii].append(k)
                if npig is None: npig=g
                else: assert npig==g
                parts.append((np.concatenate(sc), np.concatenate(im).astype(np.int64), np.concatenate(dm,1).astype(bool), np.concatenate(di,1).astype(bool), np.full(sum(len(s) for s in sc),src)))
            sc=np.concatenate([p[0] for p in parts]); im=np.concatenate([p[1] for p in parts])
            dm=np.concatenate([p[2] for p in parts],1); di=np.concatenate([p[3] for p in parts],1); sr=np.concatenate([p[4] for p in parts])
            o=np.argsort(-sc,kind='mergesort')
            self.blk.append((sc[o],im[o],dm[:,o],di[:,o],sr[o].astype(np.int8),npig))
        self.touch=[sorted(set(t)) for t in self.touch]
    def apk(self,k,sel):
        sc,im,dm,di,sr,npig=self.blk[k]
        if npig==0: return np.nan
        keep = sr==sel[im]
        m=dm[:,keep]; ig=di[:,keep]
        tp=np.cumsum(m&~ig,1).astype(float); fp=np.cumsum(~m&~ig,1).astype(float)
        R=len(self.recThrs); out=0.0
        for t in range(self.T):
            rc=tp[t]/npig; pr=tp[t]/(tp[t]+fp[t]+np.spacing(1))
            pr=np.maximum.accumulate(pr[::-1])[::-1]
            idx=np.searchsorted(rc,self.recThrs,side='left'); ok=idx<len(pr)
            q=np.zeros(R); q[ok]=pr[idx[ok]]; out+=q.mean()
        return out/self.T
    def ap(self,sel):
        sel=np.asarray(sel,np.int8)
        return np.nanmean([self.apk(k,sel) for k in range(len(self.blk))])
    def marginal(self, base_src=0):
        nI=len(self.imgIds); sel=np.full(nI,base_src,np.int8)
        base=np.array([self.apk(k,sel) for k in range(len(self.blk))])
        nvalid=np.sum(~np.isnan(base)); g=np.zeros(nI)
        for i in range(nI):
            sel[i]=1-base_src
            d=0.0
            for k in self.touch[i]:
                if np.isnan(base[k]): continue
                d+=self.apk(k,sel)-base[k]
            g[i]=d/nvalid; sel[i]=base_src
        return g
