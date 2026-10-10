"""Tile-level mixture of two detectors' outputs: detections whose centre falls in a selected tile come from the
expensive model E, the rest from M. Rules pick k of GxG letterbox tiles per image. Scored with pycocotools (AP, AP_S/M/L)."""
import sys, numpy as np, contextlib, io, time
from tiles_common import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from anchor_tiles import anchored_tiles
with contextlib.redirect_stdout(io.StringIO()): cg=COCO(D+'instances_val2017.json')
gt=gt_array(); gt=gt[gt[:,7]==0]
def ev(a):
    with contextlib.redirect_stdout(io.StringIO()):
        cd=cg.loadRes(np.column_stack([np.array(ids)[a[:,0].astype(int)],a[:,1:7]]))
        e=COCOeval(cg,cd,'bbox'); e.evaluate(); e.accumulate(); e.summarize()
    return e.stats[[0,3,4,5]]
MODELS={'M':'dump_yolo26m_coco.json','L':'dump_yolo26l_coco.json','X':'dump_yolo26x_coco.json','M768':'dumpml_yolo26m_768_coco.json','S':'dump_yolo26s_coco.json','N':'dump_yolo26n_coco.json'}
def scores(rule,G,M,E,rng):
    tg=tile_of(gt,G); rand=rng.random((N,G*G))
    if rule=='GTcount':
        c=np.zeros((N,G*G)); np.add.at(c,(gt[:,0].astype(int),tg),1); return c+1e-6*rand
    if rule.startswith('Mdet'):   # GT-free: M's own detection mass (score >= thr) per tile
        thr=float(rule[4:]); tm=tile_of(M,G); s=M[:,5]>=thr; c=np.zeros((N,G*G)); np.add.at(c,(M[s,0].astype(int),tm[s]),M[s,5]); return c+1e-6*rand
    if rule.startswith('Munc'):   # GT-free: M's uncertainty mass, sum of s(1-s) over dets with score >= thr
        thr=float(rule[4:]); tm=tile_of(M,G); s=M[:,5]>=thr; c=np.zeros((N,G*G)); np.add.at(c,(M[s,0].astype(int),tm[s]),M[s,5]*(1-M[s,5])); return c+1e-6*rand
    if rule.startswith('Mlow'):   # reverse control: the tiles with the LEAST detection mass
        thr=float(rule[4:]); tm=tile_of(M,G); s=M[:,5]>=thr; c=np.zeros((N,G*G)); np.add.at(c,(M[s,0].astype(int),tm[s]),M[s,5]); return -c+1e-6*rand
    if rule=='random': return rand
    if rule=='randomc': return rand+content_mask(G)
    raise ValueError(rule)
if __name__=='__main__':
    base,exp,G,share,rule=sys.argv[1],sys.argv[2],int(sys.argv[3]),float(sys.argv[4]),sys.argv[5]
    seed=int(sys.argv[6]) if len(sys.argv)>6 else 0
    M=load(MODELS[base])
    if exp=='-':
        t=time.time(); print(f'ALONE {base} AP/S/M/L', np.round(ev(M),4), f'{time.time()-t:.0f}s', flush=True); sys.exit()
    E=load(MODELS[exp]); rng=np.random.default_rng(seed)
    k=int(round(share*G*G)); sc=scores(rule,G,M,E,rng)
    sel=np.argsort(-sc,1)[:,:k]; S=np.zeros((N,G*G),bool); np.put_along_axis(S,sel,True,1)
    a=np.concatenate([M[~S[M[:,0].astype(int),tile_of(M,G)]], E[S[E[:,0].astype(int),anchored_tiles(M,E,G,base+exp)]]])
    st=ev(a)
    print(f'MIX base={base} exp={exp} G={G} share={share:.2f} k={k} rule={rule} seed={seed} AP={st[0]:.4f} S/M/L={st[1]:.4f}/{st[2]:.4f}/{st[3]:.4f}', flush=True)
