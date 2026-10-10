import sys; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1")
import numpy as np, mixlib as M
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco", "dump_yolo26l_coco")]; I = len(srcs[0]["imgIds"]); rng = np.random.RandomState(7)
for q, qL in ((0.6, 0.3), (0.4, 0.2)):
    v = []
    for _ in range(3):
        p = rng.permutation(I); ch = np.ones(I, int); a, b = int(round(q * I)), int(round(qL * I)); ch[p[:a]] = 0; ch[p[a:a + b]] = 2; v.append(M.score(srcs, ch)[0])
    print(f"null q512 {q} qL {qL}: {np.mean(v):.4f} draws {v}", flush=True)
