# Agent 5 (Mixture of Tokens, TCR stub): build-feasibility request, round 2 -> round 3

Build only, plus a same-session H200 timing for the order of the dispatch overhead. Not a device-of-record number.

## Files (all static batch 1, input `images` [1,3,640,640] fp32, output `output0` [1,300,6], opset 18, onnxsim'd)

| file | what it is | priority |
|---|---|---|
| `tcr_m_base.onnx` | public YOLO26-M through the same wrapper/exporter (no routing). Same-exporter comparator | 1 |
| `tcr_k16_gather.onnx` | M + router + TopK(16 of 100 tiles) -> Gather on constant tile tables -> Gather on flattened L2/L16/L19 maps -> 4 window-transformer layers (64 tokens, d 256, 8 heads, SDPA, FFN 4x, random weights) -> expert head (stride-8 + 4 stride-4 anchors per token, 5,120 anchors) -> mask of M's P3 anchors in routed tiles (Equal one-hot -> ReduceMax -> Expand) -> Concat -> the end-to-end head's TopK(300) | 1 |
| `tcr_k16_onehot.onnx` | same, dispatch as one-hot MatMul: Equal(arange(100), idx) -> Cast -> [16,100] x tile-major maps. No runtime-index Gather except the one already in M's end-to-end head | 1 |
| `tcr_k16_gather_halo2.onnx` | `gather` + a 2-token ring of L16 tokens (80 per tile, zero row for out-of-image) as extra keys/values in every layer (attention halo, no conv) | 2 |

Each has a `<name>.metadata.yaml` sidecar (copy of `yolo26m.metadata.yaml`). The comparator also in the same session:
`/data/tmp/l4-row0/onnx/yolo26m.onnx` (public export).

No ScatterElements, no NonZero, no data-dependent shape anywhere. TopK k is a constant (16; 300). Generator:
`/data/tmp/ds-yolo/seminar5/work/agent5/stub/tcr_stub.py` (`--lowering gather|onehot|gather_halo2 --mode tcr|base|off`).

## CPU checks already done (onnxruntime, 1 thread; `stub/ort_check2.log`)

- `tcr_m_base` and the `off` mode of both lowerings (expert scores forced to 0, no mask) reproduce `yolo26m.onnx` on 8
  val2017 images: every row with score > 0.05 within 1.2e-4 px / 1.1e-6 score.
- `gather` and `onehot` outputs are bit-identical on 4 images (same weights).
- With random expert weights almost all 300 output rows come from the expert (sigmoid about 0.5). That is expected and
  does not matter for build or timing.

## Please report

1. Build pass/fail per file, TensorRT 10.16.1, fp16, static, batch 1, with the error text on failure.
2. trtexec-native median of 500 iterations for `yolo26m.onnx`, `tcr_m_base`, `tcr_k16_gather`, `tcr_k16_onehot` (and
   `halo2` if it builds), all in one session.
3. Per-layer profile (`--dumpProfile --separateProfileRun`) of `tcr_k16_gather` and `tcr_k16_onehot`, kept as files.
   I will split it into router (ReduceSum/TopK), dispatch (Gather or Equal/MatMul and the transposes), expert (4 layers +
   head) and combine (mask, Concat, final TopK).

Stub expert MACs (computed): 3.94 GMAC (fuse 0.47, 4 layers x 0.84, head 0.11); the one-hot dispatch adds 0.14 GMAC
plus full-map transposes of L2 (6.6 M values), L16 and L19. The stub uses FFN 4x and so is heavier than the round-1
proposal (FFN 2x, 2.9 GMAC); its time is an upper bound for that expert.

## Kill rule this settles (K0, fixed in round 1)

Builds static, and on the H200 profile the non-expert part (router + dispatch + combine) costs <= 0.35x a dense L16
block in the same profile. If `gather` fails and `onehot` builds, the design moves to `onehot`. If both fail: kill.
