# Agent 7 request, round 2 (H200, about 1 H200 h, no training)

All ONNX here are static batch 1, exported on CPU from the public weights (`/data/yolo-quant-work/weights/yolo26{n,m,l}.pt`) by
the YOLO-Master exporter (read-only use), output [1, 300, 6] end2end. Each has a `<name>.metadata.yaml` sidecar copied from
`/data/tmp/l4-row0/onnx/yolo26m.metadata.yaml` with `imgsz` set to the engine's input size. Builder: `../r2/build_xch.py`.
CPU check: `xch_v1_f1` reproduces standalone `yolo26l_640.onnx` exactly (onnxruntime max abs diff 0.0).

## 1. Three AP dumps (protocol: multi-label, conf 0.001, IoU 0.7, max_det 300, full val2017, pycocotools)

| file | imgsz (letterbox) | what it measures | compare with |
|---|---|---|---|
| `yolo26l_544.onnx` | 544 | the dense iso-latency comparator of the exchange: public L at 544 (T4 est. 5.32 ms unset, dense M 5.36) | interpolated 0.5317; the exchange 0.5305 |
| `xch_v1_f0.onnx` | 640 | 640 letterbox -> in-engine Resize to 512 -> M@512 (the 512 path of agent 1's RS-2 and of the exchange) | M@512 letterbox dump 0.5063 |
| `xch_v2_f1.onnx` | 768 | 768 letterbox -> in-engine Resize to 640 -> L@640 (the all-Resize lowering) | L@640 dump 0.5417 |

## 2. Build and per-layer profile check (structure only; H200 times are not of record and are never compared with T4 rows)

Build fp16, TensorRT 10.16.1, static, in one session: `xch_v1_f0`, `xch_v1_f1`, `xch_v2_f0`, `xch_v2_f1`, the existing
`/data/tmp/ds-yolo/phase2/onnx/if_res2.onnx` and `if_res2_force0.onnx`, and the standalone `../r2/src/yolo26l_640.onnx` and
`../r2/src/yolo26m_512.onnx`. TensorRT-native median of 500 iterations each (real letterboxed val images if the harness has
them), plus the per-layer profile, with the time attributed inside the taken branch versus outside the conditional.

Question: on the T4, the directly-fed 768 branch of `if_res2` ran +0.65 ms over standalone (7.14 vs 6.49 real), and its profile
put 2.44 ms outside the branch where the router accounts for about 0.18 ms. Does the H200 show the same pattern for `if_res2`
and for `xch_v1_f1` (L fed directly from the engine input), and does `xch_v2_f1` (L behind a Resize) remove it? If the H200
does not reproduce the `if_res2` penalty, report that; the question then stays for the T4 session.
