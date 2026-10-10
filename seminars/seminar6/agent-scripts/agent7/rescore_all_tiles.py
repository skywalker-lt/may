"""Attack number for lens 9: X@640 re-score-only on L@512 geometry in ALL 100 tiles (the dense-distillation ceiling at
zero inference cost), using agent 9's loader and matching (read-only import). ONE thread, one pycocotools evaluation."""
import os; os.environ['OMP_NUM_THREADS'] = '1'
import sys, numpy as np
sys.path.insert(0, '/data/tmp/ds-yolo/seminar6/work/agent9')
os.chdir('/data/tmp/ds-yolo/seminar6/work/agent9')  # caches live there
from teacher_tiles import load, run, unc_route, anchored, tile_of, iou, N, G
B = load('L512'); Bi = B[:, 0].astype(int); tb = tile_of(B, G); rng = np.random.default_rng(0)
S = np.ones_like(unc_route(B, G, 16, rng), dtype=bool); inB = S[Bi, tb]
E = load('X'); Ei = E[:, 0].astype(int); te = anchored(B, E, G, 'L512X'); inE = S[Ei, te]
a = np.where(inB)[0]; b = np.where(inE)[0]
newscore = B[:, 5].copy(); matched = np.zeros(len(B), bool)
ao = a[np.argsort(Bi[a], kind='stable')]; bo = b[np.argsort(Ei[b], kind='stable')]
ab = np.searchsorted(Bi[ao], np.arange(N + 1)); bb = np.searchsorted(Ei[bo], np.arange(N + 1))
for i in range(N):
    A = ao[ab[i]:ab[i + 1]]; Bq = bo[bb[i]:bb[i + 1]]
    if len(A) == 0 or len(Bq) == 0: continue
    U = iou(B[A, 1:5], E[Bq, 1:5]) * (B[A, 6][:, None] == E[Bq, 6][None, :])
    for q in np.argsort(-E[Bq, 5]):
        c = U[:, q].argmax()
        if U[c, q] >= 0.5: newscore[A[c]] = E[Bq[q], 5]; matched[A[c]] = True; U[c, :] = -1
R = B.copy(); R[:, 5] = newscore; R[inB & ~matched, 5] *= 0.1
print(f'all tiles: base dets {inB.sum()}, matched {matched.mean():.3f}', flush=True)
run('X@640 re-score only on L@512 geometry, ALL 100 tiles', R)
