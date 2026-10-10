"""Equivalence on detections that matter: for every reference row with score > 0.05, the max |box diff| to the
same-rank row of the stub (rows sorted by score), plus the number of score rank swaps."""
import glob, numpy as np, onnxruntime as ort
from ort_check import letterbox, sess
S = {n: sess(f) for n, f in [('ref', '/data/tmp/l4-row0/onnx/yolo26m.onnx'), ('base', 'sim_base.onnx'),
     ('goff', 'sim_gather_off.onnx'), ('ooff', 'sim_onehot_off.onnx')]}
for p in sorted(glob.glob('/data/datasets/coco/images/val2017/*.jpg'))[:8]:
    x = letterbox(p); r = S['ref'].run(None, {'images': x})[0][0]; keep = r[:, 4] > 0.05
    line = []
    for n in ['base', 'goff', 'ooff']:
        o = S[n].run(None, {'images': x})[0][0]
        d = np.abs(o[keep] - r[keep]).max() if keep.any() else 0.0
        line.append(f'{n}: rows>0.05 {keep.sum()} max|diff| {d:.2e} score max|diff| {np.abs(o[:,4]-r[:,4]).max():.2e}')
    print(p.split('/')[-1], ' | '.join(line))
