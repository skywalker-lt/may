# GT-count LAND against (a) its own permuted null and (b) the whole-image GT count rule at the same average latency (BAND 64, tail costs).
import os, numpy as np
os.environ['OMP_NUM_THREADS']='1'
from mixlib import *
C=[load('dumpml_yolo26m_coco'),load('boxmix64_Lsmall'),load('boxmix64_Lmedlarge'),load('dump_yolo26l_coco')]
ids=np.array(C[0]['imgIds']); I=len(ids); g=gt(); ann=lambda i:[a for a in g.imgToAnns[int(i)] if not a['iscrowd']]
bb=lambda a:a['bbox'][2]*a['bbox'][3]
ns=np.array([sum(bb(a)<64**2 for a in ann(i)) for i in ids]); nml=np.array([sum(bb(a)>=64**2 for a in ann(i)) for i in ids])
cost=np.array([0,0.24,0.97,1.21]); A=C[0]['stats'][0]; a_s=(C[1]['stats'][0]-A)/ns.sum(); a_ml=(C[2]['stats'][0]-A)/nml.sum()
for lam in (2.5e-6,1e-6):
    G=np.stack([0*ns,a_s*ns,a_ml*nml,a_s*ns+a_ml*nml],1)-lam*cost[None]; ch=G.argmax(1); d=cost[ch].mean()
    ap=mix_ap(C,ch)[0]; nl=mix_ap(C,np.random.RandomState(7).permutation(ch))[0]
    k=int(round(d/1.21*I)); wc=np.zeros(I,int); wc[np.argsort(-(ns+nml+1e-3*np.random.RandomState(1).rand(I)))[:k]]=3; wi=mix_ap(C,wc)[0]
    print(f'lam {lam:.1e} GT LAND +{d:.3f} ms AP {ap:.4f} null {nl:.4f} ({ap-nl:+.4f}) whole-image GT count rule same latency (L share {k/I:.3f}) {wi:.4f}',flush=True)
