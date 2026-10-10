import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, onnxruntime as ort, cv2, sys
sys.path.insert(0,'/data/YOLO-Master')
from ultralytics.data.augment import LetterBox
R='/data/tmp/ds-yolo/seminar5/work/agent4/request/'
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
def sess(p): return ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
names={'official':'/data/tmp/l4-row0/onnx/yolo26m.onnx','ref':R+'yolo26m_ref640.onnx','no10':R+'yolo26m_noattn10.onnx','no10_22':R+'yolo26m_noattn10_22.onnx'}
S={k:sess(v) for k,v in names.items()}
lb=LetterBox((640,640),auto=False)
for f in ('000000000139','000000000632','000000001000'):
    im=lb(image=cv2.imread(f'/data/datasets/coco/images/val2017/{f}.jpg'))
    x=(im[...,::-1].transpose(2,0,1)[None].astype(np.float32)/255).copy()
    out={k:s.run(None,{s.get_inputs()[0].name:x})[0][0] for k,s in S.items()}
    for k,o in out.items():
        top=o[o[:,4]>0.25]; print(f, k, 'n>0.25 =',len(top), 'top5 scores', np.round(np.sort(o[:,4])[::-1][:5],3), 'maxabsdiff vs official', float(np.abs(o-out['official']).max()))
