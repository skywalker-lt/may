# Agent 2 (Mixture of Depth), round-2 request: public YOLO26-L with second units bypassed

Built on CPU from `/data/yolo-quant-work/weights/yolo26l.pt` with `/data/YOLO-Master` (dev/ds-yolo), script
`work/agent2/r2/surgery.py`. Each listed layer's `m[1]` is replaced by `nn.Identity()` (C3k2: cv2 sees [a, b, m0, m0];
C2PSA layer 10: the second PSABlock is skipped). No weight was retrained or recalibrated.

| file | units bypassed | meaning |
|---|---|---|
| yolo26l_land_b0.{pt,onnx} | 2, 4, 6, 8, 10, 13, 16, 19 | shallow branch, M topology |
| yolo26l_land_b1.{pt,onnx} | 2, 4, 6, 8, 10, 13, 19 | deep P3 |
| yolo26l_land_b2.{pt,onnx} | 2, 4, 16 | deep P4/P5 |
| yolo26l_land_b3.{pt,onnx} | 2, 4 | deep everywhere except the stem |

ONNX: static [1,3,640,640] -> [1,300,6] (end2end), opset from the fork's exporter, `simplify=True`, fp32 weights;
onnxruntime matches PyTorch to 1e-4 on boxes (`r2/parity.log`). Sidecars `<name>.metadata.yaml` copied from
`/data/tmp/l4-row0/onnx/yolo26m.metadata.yaml`.

## Priority: LOW. The CPU receipt already fires kill rule K2.

On 250 random val2017 images (CPU fp32, one thread; `r2/l_subset.log`): B0 AP **0.0019**, B3 AP **0.1790**, against
public M 0.5675 and public L 0.5892 on the same images (CPU pipeline vs TRT dump of M on these images: -0.0009).
BatchNorm recalibration on 64 train2017 images rescues the stem bypass partly but not B0 or B1 (`r2/bnrecal_probe.log`:
recall of full-L confident detections 0.75 for B3, 0.60 for B2, 0.08 for B1, 0.06 for B0).

So these branches cannot answer K1 (they are broken, not shallow). If H200 time is left at the END of the batch, dump
only b0 and b3 under the protocol (multi-label, conf 0.001, IoU 0.7, max_det 300, full val2017; about 0.2 h) as the
full-val K2 receipt. Otherwise skip: nothing in round 3 depends on it.
