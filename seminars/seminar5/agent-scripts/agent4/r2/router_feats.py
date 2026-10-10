"""N-stem-at-320 router features computed exactly as the engine will: 640 letterbox -> linear Resize to 320 -> stem_n_320.onnx
(pooled 128-d). val2017 (all 5000) and a 12,000-image random train2017 subset with YOLO-label box counts. CPU, one thread."""
import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, onnxruntime as ort, cv2, glob, sys, json
sys.path.insert(0,'/data/YOLO-Master')
from ultralytics.data.augment import LetterBox
OUT='/data/tmp/ds-yolo/seminar5/work/agent4/r2'
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
s=ort.InferenceSession('/data/tmp/ds-yolo/phase2/onnx/stem_n_320.onnx', so, providers=['CPUExecutionProvider'])
lb=LetterBox((640,640),auto=False)
def feat(path):
    im=lb(image=cv2.imread(path)); x=(im[...,::-1].transpose(2,0,1)[None].astype(np.float32)/255)
    t=cv2.resize(x[0].transpose(1,2,0),(320,320),interpolation=cv2.INTER_LINEAR).transpose(2,0,1)[None].copy()
    return s.run(None,{'images':t})[0].reshape(-1)
which=sys.argv[1]
if which=='val':
    ids=sorted(int(os.path.basename(p)[:-4]) for p in glob.glob('/data/datasets/coco/images/val2017/*.jpg'))
    F=np.stack([feat(f'/data/datasets/coco/images/val2017/{i:012d}.jpg') for i in ids]); np.savez(f'{OUT}/rfeat_val.npz',ids=np.array(ids),F=F)
else:
    labs=sorted(glob.glob('/data/datasets/coco/labels/train2017/*.txt')); rng=np.random.default_rng(0); pick=rng.choice(len(labs),12000,replace=False)
    ids=[]; F=[]; cnt=[]
    for k in pick:
        p=labs[k]; iid=os.path.basename(p)[:-4]; ip=f'/data/datasets/coco/images/train2017/{iid}.jpg'
        im=cv2.imread(ip); h,w=im.shape[:2]
        L=np.loadtxt(p,ndmin=2) if os.path.getsize(p)>0 else np.zeros((0,5))
        sz=np.sqrt(L[:,3]*w*L[:,4]*h) if len(L) else np.zeros(0)
        cnt.append([len(L),(sz<32).sum(),((sz>=32)&(sz<96)).sum(),(sz>=96).sum()])
        ids.append(int(iid)); F.append(feat(ip))
    np.savez(f'{OUT}/rfeat_train12k.npz',ids=np.array(ids),F=np.stack(F),cnt=np.array(cnt))
print('done',which)
