import sys; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1")
import mixlib as M, numpy as np
for n in ("dump_yolo26l_coco",):
    e = M.evaluated(n); print(n, M.score([e], np.zeros(len(e["imgIds"]), int)))
