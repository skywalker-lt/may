# CPU prediction of the rs2_fed engine dump: run the engine's own router subgraph (640 letterbox -> in-graph Resize 320 -> N stem ->
# ridge -> score) on every val image, route by score < thr, and score the predicted dump as the per-image mixture of the 512 / 640 dumps.
import sys; sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1"); sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent1/r2")
import numpy as np, cv2, onnxruntime as ort, mixlib as M, feats as F
from n320feat import letterbox
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
S = ort.InferenceSession("/data/tmp/ds-yolo/seminar5/work/agent1/r2/rs2_router.onnx", so, providers=["CPUExecutionProvider"])
R = np.load("/data/tmp/ds-yolo/seminar5/work/agent1/r2/router_train.npz")
ids, ns, nm, nl, _ = F.gt_counts(); Xn, _ = F.features(ids); I = len(ids)
direct = ((Xn - R["mu"]) / R["sd"]) @ R["w"] + R["b"]
eng = np.zeros(I)
for k, i in enumerate(ids):
    x = letterbox(cv2.imread(f"/data/datasets/coco/images/val2017/{int(i):012d}.jpg"), 640)[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255
    eng[k] = float(np.asarray(S.run(None, {"images": np.ascontiguousarray(x)})[0]).reshape(-1)[0])
np.save("/data/tmp/ds-yolo/seminar5/work/agent1/r2/engine_router_scores.npy", eng)
thr = float(R["thr"]); che = (eng >= thr).astype(int); chd = (direct >= thr).astype(int)
srcs = [M.evaluated(n) for n in ("dumpml_yolo26m_512_coco", "dumpml_yolo26m_coco")]
print(f"route agreement engine-router vs direct-n320 router: {np.mean(che == chd):.4f}; 512 share engine {1 - che.mean():.3f} / direct {1 - chd.mean():.3f}; score corr {np.corrcoef(eng, direct)[0,1]:.5f}")
print("predicted rs2_fed dump AP (engine route, 512/640 dumps):", M.score(srcs, che), "| direct route:", M.score(srcs, chd))
