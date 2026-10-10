"""Round 3 (agent 2): does the tile route compose with the dense envelope? Tile mixture on a reduced-scale L base.
Base B's detections in the k routed tiles (B's own uncertainty mass, sum s(1-s) over dets with s>=0.01, 10x10 tiles of the
640 letterbox, as agent 5) are swapped for expert E's when their letterbox size < thr (sizes and tiles of E taken from
their best same-class IoU>=0.5 match in B, as agent 5's mix_p3.py). Agent 5's helpers are re-implemented here so that
no file in work/agent5 is written. One CPU thread. Usage: tilemix.py BASE EXP SHARE RULE THR [SEED]"""
import os, sys, json, io, contextlib, numpy as np
os.environ['OMP_NUM_THREADS']='1'
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
D1='/data/tmp/ds-yolo/seminar5/inputs/dumps/'; D2='/data/tmp/ds-yolo/seminar5/inputs/dumps_r2/'
C='/data/tmp/ds-yolo/seminar5/work/agent2/r3/cache/'
MODELS={'M':D1+'dump_yolo26m_coco.json','L':D1+'dump_yolo26l_coco.json','X':D1+'dump_yolo26x_coco.json',
        'M768':D1+'dumpml_yolo26m_768_coco.json','M512':D1+'dumpml_yolo26m_512_coco.json',
        'L448':D2+'dumpml_yolo26l_448_coco.json','L512':D2+'dumpml_yolo26l_512_coco.json','L576':D2+'dumpml_yolo26l_576_coco.json',
        'M576':D2+'dumpml_yolo26m_576_coco.json','M608':D2+'dumpml_yolo26m_608_coco.json'}
GT=json.load(open(D1+'instances_val2017.json'))
imgs={im['id']:im for im in GT['images']}; ids=sorted(imgs); idx={i:k for k,i in enumerate(ids)}; N=len(ids)
W=np.array([imgs[i]['width'] for i in ids],float); H=np.array([imgs[i]['height'] for i in ids],float)
R=np.minimum(640/W,640/H); PX=(640-np.round(W*R))/2; PY=(640-np.round(H*R))/2
with contextlib.redirect_stdout(io.StringIO()): cg=COCO(D1+'instances_val2017.json')
def load(name):
    f=C+name+'.npy'
    if os.path.exists(f): return np.load(f)
    d=json.load(open(MODELS[name]))
    a=np.array([[idx[x['image_id']],*x['bbox'],x['score'],x['category_id']] for x in d],float); np.save(f,a); return a
def tile_of(a,G):
    i=a[:,0].astype(int); cx=(a[:,1]+a[:,3]/2)*R[i]+PX[i]; cy=(a[:,2]+a[:,4]/2)*R[i]+PY[i]
    T=640/G; return np.clip((cy//T).astype(int),0,G-1)*G+np.clip((cx//T).astype(int),0,G-1)
def iou(a,b):
    ax2=a[:,None,0]+a[:,None,2]; ay2=a[:,None,1]+a[:,None,3]; bx2=b[None,:,0]+b[None,:,2]; by2=b[None,:,1]+b[None,:,3]
    iw=np.clip(np.minimum(ax2,bx2)-np.maximum(a[:,None,0],b[None,:,0]),0,None); ih=np.clip(np.minimum(ay2,by2)-np.maximum(a[:,None,1],b[None,:,1]),0,None)
    inter=iw*ih; return inter/(a[:,None,2]*a[:,None,3]+b[None,:,2]*b[None,:,3]-inter+1e-9)
def match(M,E,tag):
    f=C+'match_'+tag+'.npy'
    if os.path.exists(f): return np.load(f)
    m=-np.ones(len(E),int); Mi=M[:,0].astype(int); Ei=E[:,0].astype(int)
    mo=np.argsort(Mi,kind='stable'); eo=np.argsort(Ei,kind='stable')
    mb=np.searchsorted(Mi[mo],np.arange(N+1)); eb=np.searchsorted(Ei[eo],np.arange(N+1))
    for i in range(N):
        a=mo[mb[i]:mb[i+1]]; b=eo[eb[i]:eb[i+1]]
        if len(a)==0 or len(b)==0: continue
        U=iou(E[b,1:5],M[a,1:5])*(E[b,6][:,None]==M[a,6][None,:]); j=U.argmax(1); ok=U.max(1)>=0.5; m[b[ok]]=a[j[ok]]
    np.save(f,m); return m
def lsize(a): i=a[:,0].astype(int); return np.sqrt(a[:,3]*a[:,4])*R[i]
def ev(a):
    with contextlib.redirect_stdout(io.StringIO()):
        cd=cg.loadRes(np.column_stack([np.array(ids)[a[:,0].astype(int)],a[:,1:7]]))
        e=COCOeval(cg,cd,'bbox'); e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[[0,3,4,5]]
if __name__=='__main__':
    base,exp,share,rule,thr=sys.argv[1],sys.argv[2],float(sys.argv[3]),sys.argv[4],float(sys.argv[5])
    seed=int(sys.argv[6]) if len(sys.argv)>6 else 0; G=int(os.environ.get('TILE_G','10'))
    M=load(base)
    if exp=='-':
        st=ev(M); print(f'ALONE {base} AP={st[0]:.4f} S/M/L={st[1]:.4f}/{st[2]:.4f}/{st[3]:.4f}',flush=True); sys.exit()
    E=load(exp); rng=np.random.default_rng(seed); k=int(round(share*G*G)); tm=tile_of(M,G)
    if rule=='random': sc=rng.random((N,G*G))
    else:
        t=float(rule[4:]); s=M[:,5]>=t; sc=np.zeros((N,G*G)); np.add.at(sc,(M[s,0].astype(int),tm[s]),M[s,5]*(1-M[s,5])); sc+=1e-6*rng.random((N,G*G))
    sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1)
    mm=match(M,E,base+'_'+exp); te=tile_of(E,G).copy(); se=lsize(E); sm=lsize(M); ok=mm>=0; te[ok]=tm[mm[ok]]; se[ok]=sm[mm[ok]]
    outM=S[M[:,0].astype(int),tm]&(sm<thr); inE=S[E[:,0].astype(int),te]&(se<thr)
    st=ev(np.concatenate([M[~outM],E[inE]]))
    print(f'TILEMIX G={G} base={base} exp={exp} share={share:.2f} rule={rule} thr={thr:g} seed={seed} swapped={outM.sum()} inserted={inE.sum()} AP={st[0]:.4f} S/M/L={st[1]:.4f}/{st[2]:.4f}/{st[3]:.4f}',flush=True)
