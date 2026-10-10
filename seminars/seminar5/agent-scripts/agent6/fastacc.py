# Fast re-accumulation of COCO AP after per-(image, class) monotone rescoring.
# Valid because a per-(image,class) monotone transform leaves COCOeval.evaluateImg's
# matching and per-(image,class) maxDet truncation unchanged; only the cross-image sort changes.
import pickle, numpy as np
class FastAcc:
    def __init__(self, path):
        D = pickle.load(open(path,'rb'))
        self.D = D
        self.catIds = D['catIds']; self.imgIds = D['imgIds']
        nI = len(self.imgIds); nA = len(D['areaRng']); nK = len(self.catIds)
        self.img_index = {i:j for j,i in enumerate(self.imgIds)}
        self.cat_index = {c:j for j,c in enumerate(self.catIds)}
        self.recThrs = D['recThrs']; self.T = len(D['iouThrs'])
        # per (cat, area): arrays
        self.blocks = {}
        recs = D['recs']
        for k in range(nK):
            for a in range(nA):
                Es = [recs[k*nA*nI + a*nI + i] for i in range(nI)]
                Es = [e for e in Es if e is not None]
                if not Es: self.blocks[(k,a)] = None; continue
                sc = np.concatenate([e[2][:100] for e in Es])
                im = np.concatenate([np.full(min(len(e[2]),100), self.img_index[e[0]]) for e in Es]).astype(np.int64)
                dtm = np.concatenate([e[3][:, :100] for e in Es], axis=1)
                dtig = np.concatenate([e[4][:, :100] for e in Es], axis=1)
                gtig = np.concatenate([e[5] for e in Es])
                npig = int(np.count_nonzero(gtig==0))
                self.blocks[(k,a)] = (sc, im, dtm.astype(bool), dtig.astype(bool), npig)
    def ap(self, factor=None, area=0, logit_add=None, per_class=False):
        """factor: array [nImg, nCat] multiplicative score factor (monotone per (img,cat)).
        logit_add: array [nImg,nCat] added to logit(score). Returns mean AP@[.5:.95] for area idx."""
        R = len(self.recThrs); aps=[]
        for k in range(len(self.catIds)):
            b = self.blocks[(k,area)]
            if b is None or b[4]==0: aps.append(np.nan); continue
            sc, im, dtm, dtig, npig = b
            s = sc
            if factor is not None: s = sc * factor[im, k]
            if logit_add is not None:
                l = np.log(np.clip(sc,1e-7,1-1e-7)) - np.log1p(-np.clip(sc,1e-7,1-1e-7)) + logit_add[im,k]
                s = 1/(1+np.exp(-l))
            inds = np.argsort(-s, kind='mergesort')
            m = dtm[:, inds]; ig = dtig[:, inds]
            tps = np.logical_and(m, ~ig); fps = np.logical_and(~m, ~ig)
            tp = np.cumsum(tps, axis=1).astype(float); fp = np.cumsum(fps, axis=1).astype(float)
            prec_t = []
            for t in range(self.T):
                rc = tp[t]/npig; pr = tp[t]/(tp[t]+fp[t]+np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                q = np.zeros(R)
                idx = np.searchsorted(rc, self.recThrs, side='left')
                ok = idx < len(pr)
                q[ok] = pr[idx[ok]]
                prec_t.append(q.mean())
            aps.append(np.mean(prec_t))
        aps = np.array(aps)
        return (np.nanmean(aps), aps) if per_class else np.nanmean(aps)
