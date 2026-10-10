# round 3: evaluate dumps from inputs/dumps or inputs/dumps_r2 once (cached evalImgs), reuse mixlib.score
import os, sys, json, pickle, contextlib, io, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1")
import mixlib as M
from pycocotools.cocoeval import COCOeval
R2 = "/data/tmp/ds-yolo/seminar5/inputs/dumps_r2"
def evaluated(name):
    pk = f"{M.CACHE}/{name}.pkl"
    if os.path.exists(pk): return pickle.load(open(pk, "rb"))
    path = f"{M.D}/{name}.json" if os.path.exists(f"{M.D}/{name}.json") else f"{R2}/{name}.json"
    G = M.gt(); dets = json.load(open(path))
    with contextlib.redirect_stdout(io.StringIO()):
        dt = G.loadRes(dets); E = COCOeval(G, dt, "bbox"); E.evaluate()
    p = E.params; C, A, I = len(p.catIds), len(p.areaRng), len(p.imgIds)
    arr = np.empty(len(E.evalImgs), dtype=object); arr[:] = E.evalImgs
    out = dict(ev=arr.reshape(C, A, I), imgIds=list(p.imgIds))
    pickle.dump(out, open(pk, "wb"), protocol=4); return out
NAMES = {"M448": "dumpml_yolo26m_448_coco", "M512": "dumpml_yolo26m_512_coco", "M576": "dumpml_yolo26m_576_coco",
         "M608": "dumpml_yolo26m_608_coco", "M640": "dumpml_yolo26m_coco", "M768": "dumpml_yolo26m_768_coco",
         "L448": "dumpml_yolo26l_448_coco", "L512": "dumpml_yolo26l_512_coco", "L576": "dumpml_yolo26l_576_coco",
         "L640": "dump_yolo26l_coco", "RS2FED": "dumpml_rs2_fed_640_coco"}
if __name__ == "__main__":
    for k, n in NAMES.items():
        e = evaluated(n); a = M.score([e], np.zeros(len(e["imgIds"]), int)); print(k, n, a, flush=True)
