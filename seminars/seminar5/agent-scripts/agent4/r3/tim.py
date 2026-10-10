import os; os.environ['OMP_NUM_THREADS']='1'
import numpy as np, onnxruntime as ort, time
so=ort.SessionOptions(); so.intra_op_num_threads=1; so.inter_op_num_threads=1
R='/data/tmp/ds-yolo/seminar5/work/agent4/request/'
for n,sz in [('yolo26m_ref640',640),('yolo26m_noattn10_22_512',512)]:
    s=ort.InferenceSession(R+n+'.onnx',so,providers=['CPUExecutionProvider'])
    x=np.random.rand(1,3,sz,sz).astype(np.float32)
    s.run(None,{s.get_inputs()[0].name:x}); t=time.time()
    for _ in range(3): s.run(None,{s.get_inputs()[0].name:x})
    print(n,(time.time()-t)/3)
