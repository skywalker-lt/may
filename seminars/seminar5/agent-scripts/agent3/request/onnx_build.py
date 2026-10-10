"""Agent 3 (seminar 5): static batch-1 ONNX graphs of the public YOLO26-M with level masks, for the protocol dumps.
  m640_noP3head_o2o : shipped one-to-one head, P3 anchors (first 6,400) removed before the end2end TopK -> [1,300,6]
  m640_noP3head_o2m : one-to-many head, P3 anchors removed, raw [1,84,2000] (xywh + class scores) for the CLI's NMS
  m640_bconst_o2o   : branch B at epoch 0 (b_init.pt: layers 14-17 gone, layer 17 = train2017 per-channel mean, folded)
The trunk is the public model unchanged (no training). Each graph is checked against PyTorch with onnxruntime.
usage: PYTHONPATH=/data/YOLO-Master python onnx_build.py <yolo26m.pt> <b_init.pt> <out dir>"""
import sys, copy, shutil
from pathlib import Path
import numpy as np, torch, yaml, onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2xywh
META = "/data/tmp/l4-row0/onnx/yolo26m.metadata.yaml"
class Masked(torch.nn.Module):
    def __init__(self, net, keep_from, o2m):
        super().__init__(); self.net, self.keep_from, self.o2m = net, keep_from, o2m
    def forward(self, x):
        L = self.net.model; y = []
        for m in L[:-1]:
            if m.f != -1: x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x); y.append(x)
        det = L[-1]; feats = [y[i] for i in det.f]
        p = det.forward_head(feats, **(det.one2many if self.o2m else det.one2one))
        dec = det._inference(p)[:, :, self.keep_from:]             # [1, 84, A'] boxes xyxy (end2end model)
        if self.o2m:
            return torch.cat([xyxy2xywh(dec[:, :4].transpose(1, 2)).transpose(1, 2), dec[:, 4:]], 1)
        return det.postprocess(dec.permute(0, 2, 1))
def export(mod, out, e2e):
    mod.eval(); x = torch.rand(1, 3, 640, 640)
    with torch.no_grad(): ref = mod(x)
    torch.onnx.export(mod, x, str(out) + ".onnx", input_names=["images"], output_names=["output0"], opset_version=18,
                      do_constant_folding=True, dynamo=False)
    meta = yaml.safe_load(open(META)); meta["end2end"] = e2e
    yaml.safe_dump(meta, open(str(out) + ".metadata.yaml", "w"), sort_keys=False)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    got = ort.InferenceSession(str(out) + ".onnx", so).run(None, {"images": x.numpy()})[0]
    print(out.name, "shape", tuple(got.shape), "max |onnx - torch| %.2e" % np.abs(got - ref.numpy()).max(), flush=True)
if __name__ == "__main__":
    m_pt, b_pt, od = sys.argv[1], sys.argv[2], Path(sys.argv[3]); od.mkdir(parents=True, exist_ok=True)
    for name, kf, o2m in [("m640_noP3head_o2o", 6400, False), ("m640_noP3head_o2m", 6400, True)]:
        net = YOLO(m_pt).model.float().eval(); net.model[-1].export = True
        export(Masked(net, kf, o2m), od / name, not o2m)
    nb = torch.load(b_pt, map_location="cpu", weights_only=False)["model"].float().eval(); nb.model[-1].export = True
    export(Masked(nb, 0, False), od / "m640_bconst_o2o", True)
