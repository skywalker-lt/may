"""Attack on direction 3's C branch proxy. C is a P2 *leaf*: P3-P5 are unchanged, so only small detections can differ.
Proxy C_thr = YOLO26-M@640's detections of size >= thr plus YOLO26-M@768's detections of size < thr (size = sqrt(w*h) on the
640 letterbox). Agent 3 proxied C by the whole 768 dump. Dense C everywhere, then C routed at share 0.2 by an m640 small-count
ridge (out of fold), GT rule and null. Exact per-image mixing. CPU, one thread."""
import os, sys, json, pickle, contextlib, io; os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent1')
import numpy as np, mixlib as M, feats as F
from pycocotools.cocoeval import COCOeval
ids, ns, nm, nl, minA = F.gt_counts(); n320, m640 = F.features(ids)
G = M.gt(); imgs = {i: G.imgs[i] for i in G.imgs}
R = {i: min(640/imgs[i]['width'], 640/imgs[i]['height']) for i in imgs}
def ev_list(dets, tag):
    pk = f'{M.CACHE}/agent4_{tag}.pkl'
    if os.path.exists(pk): return pickle.load(open(pk,'rb'))
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(G, G.loadRes(dets), 'bbox'); E.evaluate()
    p = E.params; arr = np.empty(len(E.evalImgs), dtype=object); arr[:] = E.evalImgs
    out = dict(ev=arr.reshape(len(p.catIds), len(p.areaRng), len(p.imgIds)), imgIds=list(p.imgIds)); pickle.dump(out, open(pk,'wb'), protocol=4); return out
D = M.D; A = json.load(open(f'{D}/dumpml_yolo26m_coco.json')); B = json.load(open(f'{D}/dumpml_yolo26m_768_coco.json'))
sz = lambda d: np.sqrt(d['bbox'][2]*d['bbox'][3])*R[d['image_id']]
m640e, m768 = M.evaluated('dumpml_yolo26m_coco'), M.evaluated('dumpml_yolo26m_768_coco')
rng = np.random.default_rng(0)
r_s = F.oof_ridge(m640, np.log1p(ns))
def top(score, q):
    o = np.argsort(-(score + 1e-9*rng.random(len(score)))); s = np.zeros(len(score), int); s[o[:int(round(q*len(score)))]] = 1; return s
print('M640', M.score([m640e], np.zeros(len(ids)))[:4], 'M768 whole', M.score([m768], np.zeros(len(ids)))[:4], flush=True)
for thr in (16, 32):
    C = ev_list([d for d in A if sz(d) >= thr] + [d for d in B if sz(d) < thr], f'cleaf{thr}')
    print(f'thr {thr}: dense C-leaf proxy everywhere', M.score([m640e, C], np.ones(len(ids))), flush=True)
    for q in (0.2,):
        l = M.score([m640e, C], top(r_s, q)); g = M.score([m640e, C], top(ns, q)); n = np.mean([M.score([m640e, C], top(rng.random(len(ids)), q))[0] for _ in range(3)])
        w = M.score([m640e, m768], top(r_s, q))
        print(f'   share {q}: learned m640 {l} | GT small rule {g} | null {n:.4f} | (whole-768 proxy, same router: {w})', flush=True)
