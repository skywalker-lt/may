# Where does PTQ of the YOLO26-M tail break? Q/DQ variants on the first 100 images of the 500-image subset (CPU, one thread).
import os; os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, io, contextlib, time
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent10/r2"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent10")
from pre import *; from make_qdq import make; from common import gt
from pycocotools.cocoeval import COCOeval
R = "/data/tmp/ds-yolo/seminar5/work/agent10/r2"; G = gt()
ids = json.load(open(f"{R}/ids_fp32.json"))[:100]; S = set(ids)
def ap(d):
    with contextlib.redirect_stdout(io.StringIO()):
        E = COCOeval(G, G.loadRes(d), "bbox"); E.params.imgIds = ids; E.evaluate(); E.accumulate(); E.summarize()
    return E.stats[0]
ref = {t: ap([x for x in json.load(open(f"{R}/dets_{t}.json")) if x["image_id"] in S]) for t in ["fp32", "q8all", "q8stem"]}
print("100-image reference: " + ", ".join("%s %.4f" % kv for kv in ref.items()), flush=True)
for scope, meth in [x.split(":") for x in sys.argv[1:]]:
    t0 = time.time(); dst = f"{R}/_s_{scope}_{meth}.onnx"
    with contextlib.redirect_stdout(io.StringIO()): make(ONNX, f"{R}/amax.json", scope, dst, meth)
    s = session(dst); dets = []
    for img in ids:
        x, r, pw, ph = letterbox(f"{VAL}/{img:012d}.jpg"); dets += to_coco(s.run(None, {"images": x})[0], img, r, pw, ph)
    a = ap(dets); os.remove(dst)
    print("scope %-7s amax %-8s AP %.4f  (fp32 - this = %+.4f)  %.0fs" % (scope, meth, a, ref["fp32"] - a, time.time() - t0), flush=True)
