# Agent 4 (direction 4), round-2 request: dumps and one build check, H200, no training (est. 1.0-1.3 H200 h)

All files: static batch-1 ONNX, fp32 weights (build fp16), opset 20, exported on CPU from `/data/yolo-quant-work/weights/yolo26m.pt`
with the fork at `/data/YOLO-Master` (`YOLO(pt).export(format="onnx", imgsz=..., batch=1, dynamic=False, half=False, simplify=True,
device="cpu")`), each with a `<name>.metadata.yaml` copied from `/data/tmp/l4-row0/onnx/yolo26m.metadata.yaml` (imgsz edited to 512 where
the file says 512). Output [1,300,6] end2end, same as the public engines. Build scripts: `work/agent4/r2/build_noattn.py`,
`build_p4attn.py`, `build_if_merge.py`; CPU checks: `r2/sanity.log`, `r2/sanity_if.log`.

## Priority 1: dumps under the AP protocol (multi-label, conf 0.001, IoU 0.7, max_det 300, full val2017)

| # | file | what it is | expected / used for |
|---|---|---|---|
| 1 | `yolo26m_noattn10.onnx` (640) | public M, layer 10's PSABlock with `x + attn(x)` replaced by `x` (FFN kept) | YOLO26's own per-image reliance on its P5 attention; gate K0' |
| 2 | `yolo26m_noattn10_22.onnx` (640) | the same in both PSA blocks (layer 10 and layer 22's `m[0][1]`) | same, whole attention term |
| 3 | `if_shrink_p4attn0.onnx` (640 input) | the merged conditional: router (N stem at 320, ridge fitted on 12,000 train2017 images) -> `If`: then = Resize 512 + M@512 with a zero-initialised P4 C2PSA block (boxes x 1.25), else = M@640 | build check + AP parity: must reproduce the CPU mixture of the public 512/640 dumps under the same route, 0.5228 (+-0.0005 for in-engine Resize vs letterbox-512) |
| 4 | `yolo26m_noattn10_22_512.onnx` (512) | #2 at 512 | reliance at the shrink scale |

## Priority 2: build only (dump optional)

- `yolo26m_ref640.onnx`: the unmodified export through the same path (exporter parity; CPU outputs equal the l4-row0 ONNX to 2e-4).
- `yolo26m_p4attn0_512.onnx`, `yolo26m_p4attn0_640.onnx`: M with the zero-initialised P4 block (cv2 BN gamma = beta = 0, so the block adds
  exactly 0; CPU scores equal public M@512 / M@640 exactly). If dumped, AP must equal 0.5063 / 0.5261.
- For #3, please record: TensorRT 10.16.1 fp16 build pass/fail with the error text; H200 TRT-native median of 500 for #3 (real inputs, both
  branches), `yolo26m_ref640` and `yolo26m_p4attn0_512` in one session, and the per-layer profile of `yolo26m_p4attn0_512`
  (the block's kernels: is MHA fused?). Relative numbers only; not device-of-record.

## Notes

- The router of #3 is fitted on train2017 (YOLO-format labels, target log1p(#boxes under 96 px)), threshold = train median; realised val
  share 0.508. It is the train-fitted version of direction 1's thumbnail router (val AP of the plain shrink with it: 0.5228, against 0.5230
  for the out-of-fold val router).
- Thread rule: everything ran with OMP_NUM_THREADS=1 and onnxruntime intra/inter threads 1, except the exporter's built-in simplify step,
  which opened its own onnxruntime session with default threads for a few seconds per export.
