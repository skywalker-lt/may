# Check the o2o graphs on a real val2017 image (random inputs give tied TopK scores, so rows permute).
import sys, numpy as np, torch, onnxruntime as ort, cv2
sys.path.insert(0, '/data/tmp/ds-yolo/seminar5/work/agent3/request'); sys.argv = ['x']
from onnx_build import Masked
from ultralytics import YOLO
im = cv2.imread('/data/datasets/coco/images/val2017/000000000139.jpg'); h, w = im.shape[:2]; r = 640 / max(h, w)
im = cv2.resize(im, (round(w * r), round(h * r))); pad = np.full((640, 640, 3), 114, np.uint8)
t = (640 - im.shape[0]) // 2; pad[t:t + im.shape[0], :im.shape[1]] = im
x = torch.from_numpy(pad[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
so = ort.SessionOptions(); so.intra_op_num_threads = 1
for name, net, kf in [('m640_noP3head_o2o', YOLO('/data/yolo-quant-work/weights/yolo26m.pt').model.float().eval(), 6400),
                      ('m640_bconst_o2o', torch.load('smoke/b_init.pt', weights_only=False)['model'].float().eval(), 0)]:
    net.model[-1].export = True
    with torch.no_grad(): ref = Masked(net, kf, False)(x)[0].numpy()
    got = ort.InferenceSession(f'onnx_tmp/{name}.onnx', so).run(None, {'images': x.numpy()})[0][0]
    k = (ref[:, 4] > 0.01).sum(); a = ref[np.argsort(-ref[:, 4])][:k]; b = got[np.argsort(-got[:, 4])][:k]
    print(name, 'rows with score>0.01:', k, 'max |diff| over them %.2e' % (np.abs(a - b).max() if k else 0))
