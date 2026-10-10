import time, numpy as np, mixlib as M
t=time.time()
srcs=[M.evaluated(n) for n in ("dumpml_yolo26m_512_coco","dumpml_yolo26m_coco","dumpml_yolo26m_768_coco")]
print("eval time",round(time.time()-t))
I=len(srcs[0]["imgIds"])
for k in range(3): t=time.time(); print(k, M.score(srcs, np.full(I,k)), round(time.time()-t,1),"s")
