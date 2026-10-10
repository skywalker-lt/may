import time, sys; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent10/r2")
from pre import *
s = session(ONNX); x, r, pw, ph = letterbox(VAL + "/000000000139.jpg")
t = time.time(); o = s.run(None, {"images": x}); t1 = time.time() - t
t = time.time(); o = s.run(None, {"images": x}); print("first %.2f s, second %.2f s" % (t1, time.time() - t)); print(o[0][0][:3])
