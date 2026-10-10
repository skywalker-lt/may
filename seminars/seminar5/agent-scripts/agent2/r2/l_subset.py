"""K2 receipt on CPU: untrained identity-bypass YOLO26-L (B0 all eight units, B3 stem units only) on the first 250 images of
the fixed random order, against the public dumps of M and L on the same images and the CPU-pipeline offset (M CPU vs M dump)."""
import os, sys, json, io, contextlib, numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent2')
from mixlib import gt, D
from pycocotools.cocoeval import COCOeval
g = gt(); S = sorted(json.load(open('m_ids_0_400.json'))[:250]); s = set(S)
V = {'M dump': [d for d in json.load(open(f'{D}/dumpml_yolo26m_coco.json')) if d['image_id'] in s],
     'M CPU': [d for d in json.load(open('m_full_0_400.json')) if d['image_id'] in s],
     'L dump': [d for d in json.load(open(f'{D}/dump_yolo26l_coco.json')) if d['image_id'] in s]}
for n in ('lb0', 'lb3'):
    if os.path.exists(f'{n}_full_0_250.json'): V[f'{n} CPU (bypass, untrained)'] = json.load(open(f'{n}_full_0_250.json'))
for k, dts in V.items():
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(g, g.loadRes(dts) if dts else None, 'bbox'); E.params.imgIds = S; E.evaluate(); E.accumulate(); E.summarize()
    print(f'{k:32s} AP {E.stats[0]:.4f} S/M/L {E.stats[3]:.4f}/{E.stats[4]:.4f}/{E.stats[5]:.4f}  ndet {len(dts)}', flush=True)
