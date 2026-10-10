# Agent 10 measurement request, minimal version (H200, AP only, est. 0.3-0.5 H200 h)

Public YOLO26-M at 640, ONNX `/data/tmp/l4-row0/onnx/yolo26m.onnx` (static [1,3,640,640], end-to-end [1,300,6]).
No training, no H200 timing (Hopper INT8 timing says nothing about the T4).

## Why it is now small

The CPU receipt (`work/agent10/r2/`, onnxruntime fake-quant, one thread, 500-image val2017 subset; the fp32 pipeline
reproduces the TensorRT fp16 dump on that subset to 0.0000 AP) already decided the construct:
- max-calibrated symmetric INT8 (the best of entropy / max / p99.99): delta_all 0.0220 (se 0.0024), delta_stem 0.0059,
  delta_tail 0.0161 (se 0.0022); entropy calibration destroys the head (delta_all 0.143);
- the saturation tap does not see the INT8 loss (Spearman with per-image dAP_tail +0.04-0.06; +0.009 with count removed).
What is left is to confirm on TensorRT that dense PTQ INT8 M sits under the fp16 front (a fact every T4 direction uses),
and one build-feasibility bit.

## Engines (TensorRT 10.16.1, `build_int8.py`)

| id | priority | command | output |
|---|---|---|---|
| E1 | 1 | `build_int8.py fp16 yolo26m.onnx e1.engine` | full val2017 dump via `dump_trt.py`; must reproduce the CLI's 0.5261 within 0.0005 (validates the dump script) |
| E2 | 1 | `build_int8.py explicit q8max_all.onnx e2.engine` | full val2017 dump: delta_all on TensorRT |
| E3 | 2 | `build_int8.py explicit q8max_stem.onnx e3.engine` | full val2017 dump: delta_stem, delta_tail |
| E4 | 3 | `build_int8.py explicit srp_max.onnx e4.engine` | build pass / fail and the error text only (Q/DQ inside one `If` branch; useful to anyone who wants an INT8 branch) |

Optional E5 (only if time is left): `build_int8.py implicit yolo26m.onnx e5.engine --calib-dir
/data/datasets/coco/images/train2017 --n 1000 --cache e5.cache` and its dump: TensorRT's own entropy calibrator, to see
whether it also breaks the head as my CPU entropy calibration did.

Q/DQ files: `work/agent10/r2/q8max_all.onnx`, `q8max_stem.onnx`, `srp_max.onnx` (scales from `amax.json`, method `max`,
128 train2017 images listed in `calib_images.json`). Kept in floating point: layer 0, the two PSA blocks, the six final
head convolutions. Dump command: `python dump_trt.py <engine> /data/datasets/coco/images/val2017 <out>.json`.
Also keep `<engine>.layers.json` (engine inspector) to check which layers really ran INT8.

## H200 INT8 is not T4 INT8

The quantised function is frozen in the ONNX (explicit Q/DQ, fixed scales): both devices multiply the same int8 integers
with int32 accumulation. Remaining differences are fusion choices and the precision of epilogues / re-quantisation
(fp16 vs fp32). Expected gap est. <= 0.001 AP. Treatment: H200 INT8 AP is labelled "pred. for the T4"; H200 and the CPU
reference must agree on the 500-image subset within 0.0015 (else inspect `layers.json`); in the T4 session the same ONNX is
built and dumped once (kill rule K5: gap > 0.002 voids the H200 numbers). Implicit calibration is never used as a record,
because TensorRT chooses per-layer precision by timing and the INT8 layer set can differ between Hopper and Turing.

## Files

`request/`: `build_int8.py` (fp16 / explicit / implicit builds, calibration cache -> amax), `dump_trt.py`, `make_qdq.py`,
`make_srp.py`, `calib.py` (the CPU entropy/max calibration), `pre.py`, `qsel.py`. The ONNX graphs were run in onnxruntime
(each SRP branch reproduces its standalone model to 0.0 max abs diff); `build_int8.py` and `dump_trt.py` were only
syntax- and API-checked (TensorRT 10.9 locally, no GPU on this host).
