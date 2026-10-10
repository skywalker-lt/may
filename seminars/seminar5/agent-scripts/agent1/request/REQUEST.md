# Agent 1 request, seminar 5 round 2 (for round 3)

Construct: RS-2, routed shrink, one weight set (public YOLO26-M), two rungs {512, 640}, one ONNX `If`. Static, batch 1,
input [1,3,640,640], output [1,300,6] in the 640 frame (the 512 branch multiplies boxes by 1.25 inside the graph), so the CLI
post-processing is that of a plain 640 YOLO26-M. Router: in-graph Resize 640->320, YOLO26-N layers 0-5, global pool,
ridge fitted on 24,000 train2017 images (target log1p(small+medium count)), threshold at the train quantile for 512 share 0.5.
Built by `work/agent1/r2/build_rs2.py` from `/data/tmp/ds-yolo/phase2/onnx/{_m640,yolo26m_512,stem_n_320}.onnx`.

Files: `rs2_fed.onnx` (the 640 branch is fed through a scale-1 Resize, agent 7's countermeasure to the +0.65 ms
directly-fed-branch penalty of `if_res2`), `rs2_direct.onnx` (the 640 branch reads the engine input directly). Sidecars
`rs2_fed.metadata.yaml`, `rs2_direct.metadata.yaml` (copied from `/data/tmp/l4-row0/onnx/yolo26m.metadata.yaml`, imgsz 640).

Please, in priority order (H200, TensorRT 10.16.1 fp16, static):
1. Build `rs2_fed` and `rs2_direct`: pass/fail with the error text if any.
2. One val2017 dump of `rs2_fed` at 640 under the AP protocol (multi-label, conf 0.001, IoU 0.7, max_det 300), json to
   `dumps/` as `dumpml_rs2_fed_640_coco.json`. My CPU prediction of its AP is 0.5228 (S/M/L 0.3536/0.5669/0.6879) at a realised 512 share of 0.519 (`work/agent1/r2/parity.log`); the difference
   measures in-engine resize and router-lowering parity (gate: |measured - predicted| <= 0.0005).
3. Only if the protocol permits non-record timing: TRT-native median of 500 on the H200 for both engines on two real
   inputs, one routed to each branch (`in640_shrink.npy`, `in640_keep.npy` in this folder), with dense YOLO26-M@640 re-timed in
   the same session. This is a diagnostic of whether the directly-fed penalty exists on another GPU, not a device-of-record
   number; the T4 session decides K3.
