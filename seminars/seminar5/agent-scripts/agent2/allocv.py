# Level-allocated depth, proxy from public weights: four branches per image
#   0 = M (shallow everywhere), 1 = deep P3 path (L's boxes < 32^2, M's elsewhere), 2 = deep P4/P5 path (L's boxes >= 32^2), 3 = L
# chosen per image by predicted per-band instance counts under a latency multiplier lambda.  Costs: T4 real-input est.
import os, numpy as np, pickle, sys
BAND=int(os.environ.get('BAND','64')); FEAT=os.environ.get('FEAT','m640'); CS,CML=map(float,os.environ.get('COSTS','0.24,0.97').split(',')); RC=float(os.environ.get('RCOST','0'))
print('BAND',BAND,'FEAT',FEAT,'costs',CS,CML,'router',RC)
os.environ['OMP_NUM_THREADS']='1'
from mixlib import *
W='/data/tmp/ds-yolo/seminar5/work/agent2'
C=[load('dumpml_yolo26m_coco'),load('boxmix64_Lsmall' if BAND==64 else 'boxmix_Lsmall'),load('boxmix64_Lmedlarge' if BAND==64 else 'boxmix_Lmedlarge'),load('dump_yolo26l_coco')]
ids=np.array(C[0]['imgIds']); I=len(ids)
for k,c in zip(('M','M+L<64','M+L>=64','L'),C): print(f'dense {k:8s} AP {c["stats"][0]:.4f} S/M/L {c["stats"][3]:.4f}/{c["stats"][4]:.4f}/{c["stats"][5]:.4f}')
g=gt(); ann=lambda i:[a for a in g.imgToAnns[int(i)] if not a['iscrowd']]
bb=lambda a:a['bbox'][2]*a['bbox'][3]
ns=np.array([sum(bb(a)<BAND**2 for a in ann(i)) for i in ids]); nml=np.array([sum(bb(a)>=BAND**2 for a in ann(i)) for i in ids])
z=np.load(f'{D}/val2017_stem_pooled.npz')
def oof(X,y,lam=100.,k=5,seed=0):
    rng=np.random.RandomState(seed); fold=rng.randint(0,k,len(y)); p=np.zeros(len(y))
    for f in range(k):
        tr=fold!=f; mu=X[tr].mean(0); sd=X[tr].std(0)+1e-6; Xt=(X[tr]-mu)/sd; Xv=(X[~tr]-mu)/sd
        w=np.linalg.solve(Xt.T@Xt+lam*np.eye(X.shape[1]),Xt.T@(y[tr]-y[tr].mean())); p[~tr]=Xv@w+y[tr].mean()
    return p
ps=np.expm1(np.clip(oof(z[FEAT],np.log1p(ns)),0,None)); pml=np.expm1(np.clip(oof(z[FEAT],np.log1p(nml)),0,None))
base,cs,cml=4.85+RC,CS,CML   # T4 real-input est.: shallow branch measured 4.85; deep L branch measured 6.06 = 4.85+0.24+0.97
cost=np.array([0,cs,cml,cs+cml])
A=C[0]['stats'][0]; a_s=(C[1]['stats'][0]-A)/ns.sum(); a_ml=(C[2]['stats'][0]-A)/nml.sum()
print(f'per-instance dense gain: small {a_s*1e6:.2f}e-6, med+large {a_ml*1e6:.2f}e-6')
front=lambda t:0.5261+0.0102*(t-4.85)   # front re-based on the shallow branch: no compile-effect credit
def route(es,eml,lam):
    G=np.stack([0*es,a_s*es,a_ml*eml,a_s*es+a_ml*eml],1)-lam*cost[None]; return G.argmax(1)
lams=[float(x) for x in sys.argv[1:]] or [2e-5,4e-5,6e-5,8e-5]
for lam in lams:
    for tag,(es,eml) in (('GT counts',(ns,nml)),('stem ridge counts',(ps,pml))):
        ch=route(es,eml,lam); avg=base+cost[ch].mean(); ap,_=mix_ap(C,ch); fr=np.bincount(ch,minlength=4)/I
        print(f'lam {lam:.0e} {tag:18s} shares {fr.round(3)} avg {avg:.3f} ms  AP {ap:.4f}  front {front(avg):.4f}  AP-front {ap-front(avg):+.4f}',flush=True)
    nl=[]
    for d in range(2): nl.append(mix_ap(C,np.random.RandomState(7+d).permutation(ch))[0])
    sh=(avg-base)/(cs+cml); k=int(round(sh*I)); tot=ps+pml; wc=np.zeros(I,int); wc[np.argsort(-tot)[:k]]=3
    print(f'          whole-image M/L by stem count at the same avg (L share {sh:.3f}): AP {mix_ap(C,wc)[0]:.4f}',flush=True)
    print(f'          null (permuted branches, same shares) AP {np.mean(nl):.4f} ({" ".join("%.4f"%x for x in nl)})',flush=True)
