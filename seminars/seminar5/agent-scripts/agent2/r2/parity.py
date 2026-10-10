"""ONNX (onnxruntime, 1 thread) vs PyTorch parity for the four bypass models on one val image; plus sha256 of the files."""
import sys, hashlib, numpy as np, torch, cv2, onnxruntime as ort
torch.set_num_threads(1)
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
Q = '/data/tmp/ds-yolo/seminar5/work/agent2/request'
im = LetterBox((640, 640), auto=False, scaleup=False)(image=cv2.imread('/data/datasets/coco/val2017/000000000139.jpg'))
x = torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float() / 255
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
for n in ('b0', 'b1', 'b2', 'b3'):
    net = YOLO(f'{Q}/yolo26l_land_{n}.pt').model.eval().float()
    with torch.no_grad(): t = net(x)[0][0].numpy()
    o = ort.InferenceSession(f'{Q}/yolo26l_land_{n}.onnx', so, providers=['CPUExecutionProvider']).run(None, {'images': x.numpy()})[0][0]
    k = t[:, 4] > 0.25
    print(n, 'dets>0.25 torch', k.sum(), 'onnx', (o[:, 4] > 0.25).sum(), 'max|box diff| top-20', float(np.abs(t[:20, :5] - o[:20, :5]).max()),
          'sha256', hashlib.sha256(open(f'{Q}/yolo26l_land_{n}.onnx', 'rb').read()).hexdigest()[:16], flush=True)
