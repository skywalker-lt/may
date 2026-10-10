"""Per-image route of if_shrink_p4attn0.onnx on val2017, computed by its own router subgraph (router_only.onnx) in onnxruntime,
and the CPU parity reference: exact mixture of the public 512 / 640 dumps under that route."""
import os, sys, json; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, onnxruntime as ort, cv2
sys.path.insert(0,'/data/YOLO-Master'); sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent1')
from ultralytics.data.augment import LetterBox
import mixlib as M
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
s=ort.InferenceSession('/data/tmp/ds-yolo/seminar5/work/agent4/r2/router_only.onnx', so, providers=['CPUExecutionProvider'])
lb=LetterBox((640,640),auto=False); G=M.gt(); ids=sorted(G.getImgIds()); out={}
for i in ids:
    im=lb(image=cv2.imread(f'/data/datasets/coco/images/val2017/{i:012d}.jpg')); x=(im[...,::-1].transpose(2,0,1)[None].astype(np.float32)/255).copy()
    sc,sh=s.run(None,{'images':x}); out[i]=[float(sc),bool(sh)]
json.dump(out,open('/data/tmp/ds-yolo/seminar5/work/agent4/request/if_shrink_p4attn0.routes_val2017.json','w'))
sh=np.array([out[i][1] for i in ids]); m512,m640=M.evaluated('dumpml_yolo26m_512_coco'),M.evaluated('dumpml_yolo26m_coco')
print('engine-router shrink share', sh.mean(), 'CPU parity reference AP/S/M/L', M.score([m512,m640],np.where(sh,0,1)))
