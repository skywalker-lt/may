"""Export the Phase 1 ONNX graphs (local tool): dense reference and weight-bank variants, static 1x3x640x640."""
import collections, shutil, sys
import numpy as np, onnx, onnxruntime as ort, torch
from ultralytics import YOLO
from ultralytics.nn.modules.moe.weight_bank import bank_modules

OUT = "/data/tmp/ds-yolo/phase1/onnx"
WB = "/data/tmp/ds-yolo/weights/yolo26m-coco-wb-init.pt"
DENSE = "/data/yolo-quant-work/weights/yolo26m.pt"
jobs = [("yolo26m_ref", DENSE, None, None, True), ("yolo26m_ref_nms", DENSE, None, None, False)]
jobs += [(f"wb_{name}_{low}", WB, k, low, True) for name, k in (("top2", 2), ("top1", 1), ("soft", 4)) for low in ("conv", "matmul")]
jobs += [(f"wb_{name}_{low}", WB, k, low, True) for name, k in (("top2", 2), ("top1", 1)) for low in ("conv_local", "matmul_local")]
only = set(sys.argv[1:])
import cv2
from ultralytics.data.augment import LetterBox
_im = LetterBox((640, 640), auto=False)(image=cv2.imread("/data/datasets/coco/images/val2017/000000000139.jpg"))
x = torch.from_numpy(_im[..., ::-1].transpose(2, 0, 1).copy()).float().div(255)[None]
for name, w, k, low, e2e in jobs:
    if only and name not in only:
        continue
    y = YOLO(w)
    ref_model = y.model.float().eval()
    if k is not None:
        st = bank_modules(ref_model)[0].state
        st.top_k, st.lowering = k, None
    if not e2e:
        ref_model.model[-1].end2end = False
    with torch.no_grad():
        ref = ref_model(x); ref = (ref[0] if isinstance(ref, (list, tuple)) else ref).numpy()
    if k is not None:
        st.lowering = low
    f = y.export(format="onnx", imgsz=640, batch=1, dynamic=False, half=False, simplify=True, device="cpu", verbose=False)
    dst = f"{OUT}/{name}.onnx"; shutil.move(f, dst)
    m = onnx.load(dst); ops = collections.Counter(n.op_type for n in m.graph.node)
    so = ort.SessionOptions(); so.intra_op_num_threads = 8
    sess = ort.InferenceSession(dst, so, providers=["CPUExecutionProvider"])
    o = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]
    # end2end rows: compare the score-sorted top rows (row order can flip on ties); raw rows: compare everything
    d = np.abs(o - ref).max() if o.shape == ref.shape else float("nan")
    kept = int((ref[0, :, 4] > 0.25).sum()) if e2e else -1
    dk = np.abs(o[0, :kept] - ref[0, :kept]).max() if e2e and kept else float("nan")
    dyn_conv = sum(1 for n in m.graph.node if n.op_type == "Conv" and n.input[1] not in {i.name for i in m.graph.initializer})
    print(f"{name}: opset {m.opset_import[0].version} out {o.shape} | ORT vs torch max abs diff {d:.3e} (rows with conf>0.25: {kept}, diff {dk:.3e}) | "
          f"Conv {ops['Conv']} (kernel-as-input {dyn_conv}) MatMul {ops['MatMul']} Gather {ops['Gather']} Slice {ops['Slice']} TopK {ops['TopK']} "
          f"ArgMax {ops['ArgMax']} If {ops['If']} NonZero {ops['NonZero']} | {round(len(m.SerializeToString()) / 1e6, 1)} MB", flush=True)
