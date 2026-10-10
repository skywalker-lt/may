"""Rung-private BatchNorm by recalibration (no gradient, so the fork's recipe cannot drift it): public YOLO26-M, BN running
stats re-estimated (cumulative average, batch 8) on N random train2017 images letterboxed to S; saved as an ultralytics
checkpoint for a protocol dump at S. Run at S=512 (the shrink rung) and S=640 (control: does plain-image recal move 640?)."""
import os, sys, time, copy, numpy as np, cv2, torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(__file__)); from n320feat import letterbox
S, N = int(sys.argv[1]), int(sys.argv[2]); B = 8
ck = torch.load("/data/yolo-quant-work/weights/yolo26m.pt", map_location="cpu", weights_only=False)
net = ck["model"].float()
bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
for m in bns: m.reset_running_stats(); m.momentum = None
net.train()
imgs = sorted(os.listdir("/data/datasets/coco/images/train2017")); sel = np.random.RandomState(1).choice(len(imgs), N, replace=False)
t0 = time.time()
with torch.no_grad():
    for k in range(0, N, B):
        x = np.stack([letterbox(cv2.imread(f"/data/datasets/coco/images/train2017/{imgs[j]}"), S)[..., ::-1].transpose(2, 0, 1) for j in sel[k:k + B]])
        net(torch.from_numpy(np.ascontiguousarray(x)).float() / 255)
        if k % 128 == 0: print(S, k, round(time.time() - t0), "s", flush=True)
net.eval(); ck2 = dict(ck); ck2["model"] = net.half(); ck2["ema"] = None
torch.save(ck2, f"/data/tmp/ds-yolo/seminar5/work/agent1/request/yolo26m_bnrecal{S}.pt"); print("saved", round(time.time() - t0), "s")
