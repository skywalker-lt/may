"""Frozen-trunk risk check: is public YOLO26-M's uncertainty mass (the router statistic, sum of p(1-p) over its
top-300 rows with p >= 0.01; and the count of rows with p >= 0.25) lower on train2017 images it was trained on than on
val2017? 200 random images each, reference yolo26m.onnx, onnxruntime CPU 1 thread."""
import sys, glob, numpy as np
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent5/stub')
from ort_check import letterbox, sess
s = sess('/data/tmp/l4-row0/onnx/yolo26m.onnx'); rng = np.random.default_rng(0)
for split in ['val2017', 'train2017']:
    fs = sorted(glob.glob(f'/data/datasets/coco/images/{split}/*.jpg')); fs = [fs[i] for i in rng.choice(len(fs), 200, replace=False)]
    U, C, P = [], [], []
    for f in fs:
        p = s.run(None, {'images': letterbox(f)})[0][0][:, 4]
        q = p[p >= 0.01]; U.append((q*(1-q)).sum()); C.append((p >= 0.25).sum()); P.append(p[p >= 0.25].mean() if (p >= 0.25).any() else np.nan)
    U, C, P = map(np.array, (U, C, P))
    print(f'{split}: mean uncertainty mass {U.mean():.3f} (median {np.median(U):.3f}), dets>=0.25 {C.mean():.2f}, '
          f'mean score of dets>=0.25 {np.nanmean(P):.3f}, mass per det>=0.25 {U.sum()/max(C.sum(),1):.3f}', flush=True)
