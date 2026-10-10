# CPU, one thread: run attention-ablated and reference YOLO26-M ONNX on a fixed val2017 subset (every 5th sorted id).
import os, sys, json, time; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, onnxruntime as ort, cv2
sys.path.insert(0,'/data/YOLO-Master')
from ultralytics.data.augment import LetterBox
R='/data/tmp/ds-yolo/seminar5/work/agent4/request/'
OUT='/data/tmp/ds-yolo/seminar5/work/agent4/r3/'
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
coco91=[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
ids=sorted(int(f[:-4]) for f in os.listdir('/data/datasets/coco/images/val2017') if f.endswith('.jpg'))[::5]
jobs=[(a,int(b)) for a,b in (s.split(':') for s in sys.argv[1:])]
for name,sz in jobs:
    s=ort.InferenceSession(R+name+'.onnx',so,providers=['CPUExecutionProvider']); inp=s.get_inputs()[0].name
    lb=LetterBox((sz,sz),auto=False); dets=[]; t=time.time()
    for k,i in enumerate(ids):
        im0=cv2.imread(f'/data/datasets/coco/images/val2017/{i:012d}.jpg'); h,w=im0.shape[:2]
        im=lb(image=im0); x=(im[...,::-1].transpose(2,0,1)[None].astype(np.float32)/255).copy()
        o=s.run(None,{inp:x})[0][0]
        g=min(sz/h,sz/w); pw=(sz-round(w*g))/2; ph=(sz-round(h*g))/2
        for x1,y1,x2,y2,sc,c in o:
            if sc<0.001: continue
            x1=(x1-pw)/g; x2=(x2-pw)/g; y1=(y1-ph)/g; y2=(y2-ph)/g
            x1=min(max(x1,0),w); x2=min(max(x2,0),w); y1=min(max(y1,0),h); y2=min(max(y2,0),h)
            dets.append({'image_id':i,'category_id':coco91[int(c)],'bbox':[round(float(x1),3),round(float(y1),3),round(float(x2-x1),3),round(float(y2-y1),3)],'score':round(float(sc),5)})
        if k%100==0: print(name,k,round(time.time()-t,1),flush=True)
    json.dump(dets,open(OUT+f'sub_{name}.json','w')); print('done',name,len(dets),round(time.time()-t,1),flush=True)
