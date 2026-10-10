# ORT check of rs2_fed / rs2_direct on two val images (one routed to each rung) against the plain 640 and 512 graphs
import sys, numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1/r2"); from n320feat import letterbox
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
sess = lambda p: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"])
Rt = sess("/data/tmp/ds-yolo/seminar5/work/agent1/r2/rs2_router.onnx"); thr = float(np.load("/data/tmp/ds-yolo/seminar5/work/agent1/r2/router_train.npz")["thr"])
P = "/data/tmp/ds-yolo/phase2/onnx"; m640, m512 = sess(f"{P}/_m640.onnx"), sess(f"{P}/yolo26m_512.onnx")
prep = lambda im, n: np.ascontiguousarray(letterbox(im, n)[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255)
for fid in ("000000000139", "000000000632", "000000000724", "000000000785"):
    im = cv2.imread(f"/data/datasets/coco/images/val2017/{fid}.jpg"); x = prep(im, 640); s = float(np.asarray(Rt.run(None, {"images": x})[0]).reshape(-1)[0])
    print(fid, "router score", round(s, 4), "-> shrink" if s < thr else "-> keep 640", flush=True)
picked = {}
for fid in ("000000000139", "000000000632", "000000000724", "000000000785"):
    im = cv2.imread(f"/data/datasets/coco/images/val2017/{fid}.jpg"); x = prep(im, 640); s = float(np.asarray(Rt.run(None, {"images": x})[0]).reshape(-1)[0])
    picked.setdefault("shrink" if s < thr else "keep", (fid, x, im))
for k, (fid, x, im) in picked.items():
    np.save(f"/data/tmp/ds-yolo/seminar5/work/agent1/request/in640_{k}.npy", x)
    for v in ("fed", "direct"):
        y = sess(f"/data/tmp/ds-yolo/seminar5/work/agent1/request/rs2_{v}.onnx").run(None, {"images": x})[0]
        ref = m640.run(None, {"images": x})[0] if k == "keep" else m512.run(None, {"images": cv2.resize(x[0].transpose(1, 2, 0), (512, 512), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)[None].copy()})[0] * np.array([1.25] * 4 + [1, 1], np.float32)
        print(k, fid, v, "max |diff| vs reference rung", float(np.abs(y - ref).max()), "top det", np.round(y[0, 0], 2), flush=True)
