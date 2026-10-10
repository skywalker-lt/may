# Public YOLO26-M on CPU (one thread), three level variants from ONE trunk pass per image:
#   A    : the shipped model (all 8,400 anchors) -- checks parity with the protocol dump
#   H    : P3 head dropped (P3 anchors removed from the end2end TopK); trunk unchanged (exact, no training)
#   B0   : untrained B branch: layer 17's output replaced by a per-channel constant (its mean on train2017
#          calibration images), layers 18-22 recomputed, P4/P5 anchors only. This is B at epoch 0.
# usage: levelvar.py calib <n>            -> writes c17.pt (per-channel mean of layer-17 output on train2017)
#        levelvar.py run <start> <stop>   -> writes lv_<start>_<stop>.json.gz (detections per variant)
import os, sys, json, gzip, time
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np, torch, cv2
torch.set_num_threads(1)
sys.path.insert(0, '/data/YOLO-Master')
from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import xyxy2xywh
W = '/data/tmp/ds-yolo/seminar5/work/agent3/r2/'
COCO_IDS = [1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90]
net = YOLO('/data/yolo-quant-work/weights/yolo26m.pt').model.float().eval()
for p in net.parameters(): p.requires_grad_(False)
L = list(net.model); det = L[-1]
def letterbox(im, S=640):
    h, w = im.shape[:2]; r = min(S / h, S / w); nh, nw = int(round(h * r)), int(round(w * r))
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nh, nw) != (h, w) else im
    top, left = (S - nh) // 2, (S - nw) // 2   # ultralytics LetterBox(center=True) rounding below
    dh, dw = (S - nh) / 2, (S - nw) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1)); left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return im, r, left, top
def trunk(x, upto=None, override=None):
    y = []
    for m in L[:-1]:
        if m.f != -1: x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
        x = override[m.i] if (override and m.i in override) else m(x)
        y.append(x)
        if upto is not None and m.i == upto: break
    return y
def head_out(feats, keep_from):
    """one-to-one head decoded; anchors before keep_from removed; returns [N,6] in input px (x1y1x2y2, score, cls)."""
    p = det.forward_head(feats, **det.one2one)
    dec = det._inference(p)                               # [1, 4+nc, A]
    pr = dec.permute(0, 2, 1)[:, keep_from:]              # drop P3 anchors if keep_from=6400
    out = det.postprocess(pr)[0]                          # TopK as in the shipped end2end head
    return out[out[:, 4] >= 0.001]
def head_o2m(feats, keep_from):
    """one-to-many head + multi-label NMS (conf 0.001, IoU 0.7, max_det 300), anchors before keep_from removed."""
    p = det.forward_head(feats, **det.one2many)
    dec = det._inference(p)[:, :, keep_from:].clone()     # [1, 4+nc, A], boxes xyxy (end2end model)
    dec[:, :4] = xyxy2xywh(dec[:, :4].transpose(1, 2)).transpose(1, 2)
    return non_max_suppression(dec, conf_thres=0.001, iou_thres=0.7, multi_label=True, max_det=300, max_time_img=1e4)[0]
@torch.no_grad()
def run_img(path, c17, do_b0=True):
    im0 = cv2.imread(path); h0, w0 = im0.shape[:2]
    im, r, left, top = letterbox(im0)
    x = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
    y = trunk(x)
    res = {}
    for name, feats, kf in [('A', [y[16], y[19], y[22]], 0), ('H', [y[16], y[19], y[22]], 6400)]:
        res[name] = head_out(feats, kf)
        res[name + 'm'] = head_o2m(feats, kf)
    if not do_b0:
        res['B0'] = res['H'][:0]
    # B0: rerun 18..22 with layer 17 replaced by the constant (same shape as y[17])
    const = c17.view(1, -1, 1, 1).expand_as(y[17]) if do_b0 else None
    if do_b0:
      const = c17.view(1, -1, 1, 1).expand_as(y[17])
      yb = list(y[:17]) + [const]
      for m in L[18:-1]:
        xin = yb[m.f] if isinstance(m.f, int) else [yb[-1] if j == -1 else yb[j] for j in m.f]
        yb.append(m(xin))
      res['B0'] = head_out([y[16], yb[19], yb[22]], 6400)
    out = {}
    for k, o in res.items():
        o = o.numpy().astype(np.float64); b = o[:, :4].copy()
        b[:, [0, 2]] = ((b[:, [0, 2]] - left) / r).clip(0, w0); b[:, [1, 3]] = ((b[:, [1, 3]] - top) / r).clip(0, h0)
        out[k] = [[round(float(b[i, 0]), 2), round(float(b[i, 1]), 2), round(float(b[i, 2] - b[i, 0]), 2),
                   round(float(b[i, 3] - b[i, 1]), 2), round(float(o[i, 4]), 5), COCO_IDS[int(o[i, 5])]] for i in range(len(o))]
    return out
if __name__ == '__main__':
    if sys.argv[1] == 'calib':
        n = int(sys.argv[2]); files = sorted(os.listdir('/data/datasets/coco/images/train2017'))
        rng = np.random.default_rng(0); pick = rng.choice(len(files), n, replace=False)
        acc = None; t0 = time.time()
        with torch.no_grad():
            for j, i in enumerate(pick):
                im, r, l, t = letterbox(cv2.imread('/data/datasets/coco/images/train2017/' + files[i]))
                x = torch.from_numpy(im[:, :, ::-1].copy()).permute(2, 0, 1)[None].float() / 255
                v = trunk(x, upto=17)[17].mean(dim=(0, 2, 3))
                acc = v if acc is None else acc + v
        torch.save(acc / n, W + 'c17.pt'); print('calib done', n, f'{time.time()-t0:.0f}s')
    else:
        a, b = int(sys.argv[2]), int(sys.argv[3]); st = int(sys.argv[4]) if len(sys.argv) > 4 else 1; c17 = torch.load(W + 'c17.pt')
        ids = [int(l.strip()) for l in open(W + 'val_ids.txt')][a:b:st]
        recs = {}; t0 = time.time()
        for j, iid in enumerate(ids):
            recs[iid] = run_img(f'/data/datasets/coco/images/val2017/{iid:012d}.jpg', c17, do_b0=j % 4 == 0)
            if j % 50 == 0: print(j, f'{(time.time()-t0)/(j+1):.2f}s/img', flush=True)
        with gzip.open(W + f'lv_{a}_{b}_{st}.json.gz', 'wt') as f: json.dump(recs, f)
        print('done', len(ids), f'{time.time()-t0:.0f}s', flush=True)
