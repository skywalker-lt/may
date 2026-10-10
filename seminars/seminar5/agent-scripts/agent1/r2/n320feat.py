# n320 router feature (YOLO26-N layers 0-5 at 320, global-pooled) via onnxruntime, one thread; letterbox as ultralytics (pad 114, centred)
import os, sys, numpy as np, cv2, onnxruntime as ort
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
S = ort.InferenceSession("/data/tmp/ds-yolo/phase2/onnx/stem_n_320.onnx", so, providers=["CPUExecutionProvider"])
def letterbox(im, n):
    h, w = im.shape[:2]; r = min(n / h, n / w); nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w): im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (n - nh) // 2; left = (n - nw) // 2
    out = np.full((n, n, 3), 114, np.uint8); out[top:top + nh, left:left + nw] = im; return out
def feat(path, mode="direct"):
    im = cv2.imread(path)
    x = letterbox(im, 320) if mode == "direct" else cv2.resize(letterbox(im, 640), (320, 320), interpolation=cv2.INTER_LINEAR)
    x = (x[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255)
    return S.run(None, {"images": np.ascontiguousarray(x)})[0].reshape(-1)
if __name__ == "__main__":
    d = np.load("/data/tmp/ds-yolo/seminar5/inputs/dumps/val2017_stem_pooled.npz"); ids = d["image_id"][:40]; ref = d["n320"][:40]
    for mode in ("direct", "via640"):
        f = np.stack([feat(f"/data/datasets/coco/images/val2017/{int(i):012d}.jpg", mode) for i in ids])
        cos = (f * ref).sum(1) / np.linalg.norm(f, axis=1) / np.linalg.norm(ref, axis=1)
        print(mode, "cos min/mean", cos.min().round(5), cos.mean().round(5), "max abs diff", np.abs(f - ref).max().round(4))
