import sys, os, json, numpy as np, cv2, torch
sys.path.insert(0, '/data/YOLO-Master'); torch.set_num_threads(1)
W = '/data/yolo-quant-work/weights/yolo26m.pt'
VAL = '/data/datasets/coco/images/val2017'; TR = '/data/datasets/coco/images/train2017'
GT = '/data/tmp/ds-yolo/seminar6/inputs/dumps/instances_val2017.json'
OUT = '/data/tmp/ds-yolo/seminar6/work/agent4'
def letterbox(im, n=640):
    h, w = im.shape[:2]; r = min(n / h, n / w); nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w): im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (n - nh) // 2; left = (n - nw) // 2
    out = np.full((n, n, 3), 114, np.uint8); out[top:top + nh, left:left + nw] = im
    return out, (r, left, top)
def to_tensor(lb):
    return torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))).float()[None] / 255
def load_unfused():
    ck = torch.load(W, map_location='cpu', weights_only=False)
    net = ck['model'].float().eval()
    for p in net.parameters(): p.requires_grad_(False)
    return net
def bn_layers(net):
    return [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
def get_bn_state(net):
    return [(m.running_mean.clone(), m.running_var.clone()) for m in bn_layers(net)]
def set_bn_state(net, st):
    for m, (mu, var) in zip(bn_layers(net), st): m.running_mean.copy_(mu); m.running_var.copy_(var)
def dets_to_coco(y, iid, geo, cocoid):
    r, pw, ph = geo; d = y[0].numpy(); out = []
    for x1, y1, x2, y2, s, c in d:
        if s < 0.001: continue
        X1 = (x1 - pw) / r; Y1 = (y1 - ph) / r; X2 = (x2 - pw) / r; Y2 = (y2 - ph) / r
        out.append({'image_id': int(iid), 'category_id': cocoid[int(c)], 'bbox': [float(X1), float(Y1), float(X2 - X1), float(Y2 - Y1)], 'score': float(s)})
    return out
