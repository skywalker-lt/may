import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, onnxruntime as ort, cv2, sys
sys.path.insert(0,'/data/YOLO-Master')
from ultralytics.data.augment import LetterBox
R='/data/tmp/ds-yolo/seminar5/work/agent4/request/'
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
S=lambda p: ort.InferenceSession(p, so, providers=['CPUExecutionProvider'])
sif=S(R+'if_shrink_p4attn0.onnx'); s512=S(R+'yolo26m_p4attn0_512.onnx'); o512=S('/data/tmp/ds-yolo/phase2/onnx/yolo26m_512.onnx'); s640=S(R+'yolo26m_ref640.onnx'); rt=S('/data/tmp/ds-yolo/seminar5/work/agent4/r2/router_only.onnx')
p640=S(R+'yolo26m_p4attn0_640.onnx')
def inp(f,sz):
    im=LetterBox((sz,sz),auto=False)(image=cv2.imread(f'/data/datasets/coco/images/val2017/{f}.jpg')); return (im[...,::-1].transpose(2,0,1)[None].astype(np.float32)/255).copy()
def top(o,k=5): o=o[0]; o=o[np.argsort(-o[:,4])][:k]; return np.round(o,1)
for f in ('000000000139','000000000632','000000001000','000000000285'):
    x=inp(f,640); x5=inp(f,512)
    sc,shr=rt.run(None,{'images':x}); y=sif.run(None,{'images':x})[0]
    a=s512.run(None,{'images':x5})[0]; b=o512.run(None,{'images':x5})[0]; c=s640.run(None,{'images':x})[0]; d=p640.run(None,{'images':x})[0]
    print(f, 'router score %.3f shrink=%s'%(float(sc),bool(shr)), '| p4attn0_512 vs public 512 maxdiff %.4g'%np.abs(np.sort(a[0,:,4])-np.sort(b[0,:,4])).max(), '| p4attn0_640 vs ref640 %.4g'%np.abs(np.sort(d[0,:,4])-np.sort(c[0,:,4])).max())
    ref=c if not shr else a*np.array([1.25,1.25,1.25,1.25,1,1],np.float32)
    print('   If top-3', top(y,3).tolist()); print('   ref top-3', top(ref,3).tolist())
