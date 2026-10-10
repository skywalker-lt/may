# DS-YOLO Phases 0 and 1: T4 receipts (2026-10-06)

Device and protocol for every number below: one Tesla T4 (driver 595.91.07), yolo-master-edge 1.2.0 `trt10-cuda12`
bundle unchanged (43dfa9e), TensorRT 10.16.1 fp16, static 1x3x640x640, batch 1. Latency is the TensorRT-native
median of 500 iterations after 200 warm-up (`trt_time.py`), five rounds per batch with the baseline YOLO26-M engine
re-timed at the start of every round; the table gives the median of the five round medians. Accuracy is full
val2017 (5,000 images), the CLI's val-protocol dump (`--conf 0.001 --iou 0.7 --multi-label`) scored with
pycocotools. No number here comes from another GPU or from the CPU box.

## 1. Result

- The signed design (static weight bank for the 39 1x1 convolutions of layers 6-22, kernel-input convolution) runs
  at 1.61-1.66x YOLO26-M. Its MatMul fallback runs at 1.11-1.17x. Neither is inside the 1.05x cap.
- Residual experts (shared static kernel plus routed deltas) on all 39 layers run at 1.10-1.26x when the delta is
  applied after the static convolution, and 1.65x when the delta is summed into the kernel.
- A conditional top-1 engine (shared stem, one router, a nested ONNX `If` with four static copies of layers 6-23)
  runs at 0.966x YOLO26-M. It is the only routed variant inside the cap.
- Every routed engine built from YOLO26-M's weights reproduces its AP at epoch 0 (0.5257-0.5261 against 0.5262).

## 2. Latency table

Baseline YOLO26-M engine in the same rounds: 5.361-5.366 ms (round medians span 5.348-5.373).
"Same exporter" is YOLO26-M exported by the same code path as the routed graphs (5.507 ms).

| engine | routed layers | median ms | vs baseline | vs same exporter |
|---|---|---|---|---|
| YOLO26-M, baseline engine (table row of 2026-10-02: 5.36) | - | 5.366 | 1.000 | 0.974 |
| YOLO26-M, same exporter | - | 5.507 | 1.026 | 1.000 |
| YOLO26-M, same exporter, second build | - | 5.481 | 1.021 | 0.995 |
| YOLO26-M, end2end=False | - | 5.321 | 0.992 | 0.966 |
| **conditional top-1, four static branches** | 39 | **5.181** | **0.966** | **0.941** |
| residual scale+shift, block outputs only | 8 | 5.721 | 1.067 | 1.039 |
| residual rank-16, block outputs only | 8 | 5.861 | 1.093 | 1.064 |
| residual shift | 39 | 5.922 | 1.104 | 1.075 |
| weight bank top-1, MatMul, per-layer gather | 39 | 5.953 | 1.110 | 1.081 |
| residual scale+shift | 39 | 6.044 | 1.126 | 1.098 |
| weight bank top-1, MatMul, one gather | 39 | 6.048 | 1.127 | 1.098 |
| weight bank top-2, MatMul, per-layer gather | 39 | 6.130 | 1.143 | 1.113 |
| weight bank top-2, MatMul, one gather | 39 | 6.222 | 1.160 | 1.130 |
| weight bank soft (k = E), MatMul | 39 | 6.258 | 1.166 | 1.136 |
| residual rank-16 | 39 | 6.759 | 1.260 | 1.227 |
| weight bank top-1, kernel-input conv, per-layer gather | 39 | 8.040 | 1.500 | 1.460 |
| weight bank top-2, kernel-input conv, per-layer gather | 39 | 8.429 | 1.572 | 1.531 |
| weight bank top-1, kernel-input conv, one gather | 39 | 8.646 | 1.611 | 1.570 |
| weight bank top-2, kernel-input conv, one gather (the signed lowering) | 39 | 8.763 | 1.633 | 1.591 |
| residual full-rank, summed in weight space | 39 | 8.877 | 1.654 | 1.612 |
| weight bank soft, kernel-input conv | 39 | 8.886 | 1.656 | 1.614 |

Residual variants are timing prototypes with small random deltas; the weight-bank and conditional engines carry
YOLO26-M's weights plus 1% zero-sum perturbations and a calibrated, untrained router.

## 3. Accuracy at epoch 0 (full val2017)

| engine | AP | AP50 | AP_S | AP_M | AP_L | CLI per-frame infer |
|---|---|---|---|---|---|---|
| YOLO26-M, same exporter | 0.5262 | 0.7058 | 0.3587 | 0.5699 | 0.6874 | 4.679 ms |
| conditional top-1 | 0.5257 | 0.7053 | 0.3567 | 0.5695 | 0.6865 | 4.447 ms |
| weight bank top-1, MatMul | 0.5260 | 0.7055 | 0.3574 | 0.5694 | 0.6867 | 5.144 ms |
| weight bank top-2, MatMul | 0.5261 | 0.7056 | 0.3578 | 0.5698 | 0.6866 | 5.345 ms |

The gate was |dAP| <= 0.001 against the dense model: pass. The last column is the CLI's own timer over the 5,000
real images (a different timer from section 2; compare ratios only). It exercises all four branches of the
conditional engine, which stays 5% faster than YOLO26-M.

## 4. What the profiles say

- **Fusion was not the failure.** A convolution whose kernel is a runtime tensor keeps its fused activation (81
  fused Conv+PWN lines), but the convolutions themselves cost 5.26 ms against 3.52 ms static, and preparing the
  kernels (slice, reshape, cast) costs another 2.6 ms. Micro chains confirm it: eight 1x1 layers take 0.354 ms
  static and 0.993 ms with runtime kernels; eight 3x3 layers 0.593 against 1.301 ms.
- **The cost of routing is per routed layer, not per byte.** Any runtime tensor entering a layer stops the
  convolution fusing with its activation: about 0.014 ms per layer for a shift, 0.017 ms for scale and shift,
  0.036 ms for rank-16, 0.015-0.02 ms for a MatMul bank. Soft mixing (reads all four experts) is only 0.04 ms
  slower than top-2, so "reads k/E of the weight bytes" has no measurable latency value on the T4.
- **The conditional engine executes one branch.** Its profile shows 92 layers and 2.40 ms in the taken branch and
  0.000 ms in the other three; the router costs 0.005 ms. The engine is 82 MB (YOLO26-M: 41 MB).
- **The exporter moves YOLO26-M by 2.6%.** The same weights exported on this branch run at 5.48-5.51 ms because the
  end2end post-processing lowers to a slower TopK (0.258 ms against 0.085 ms in the baseline engine). Routed
  graphs exported on this branch carry the same penalty, so "vs same exporter" is the like-for-like column.

## 5. Gates of the plan

| gate | result |
|---|---|
| epoch-0 AP within 0.001 of the dense model | pass (0.5257-0.5261 vs 0.5262) |
| step-0 router logit std >= 0.1 | pass: 0.169 between images, 0.050 for an image and its flip (after adding parameter-free standardisation of the pooled feature; default init gave 0.011) |
| shuffle control | not usable as worded: a permutation cannot change a standard deviation, and pixel-shuffled images raise it (0.31). Replaced by the flip comparison above |
| fused Conv+PWN lines kept | kernel-input conv: yes (81); MatMul lowering: replaced by fused FC+bias+SiLU kernels |
| router plus gather <= 0.16 ms | conditional: 0.005 ms; weight bank top-2: 0.19-0.25 ms (fail) |
| L inside the cap (1.05x) | static weight bank: fail in every lowering; residual on 39 layers: fail; conditional top-1: pass |
| cap as voted, min(1.05x, median + 2 sigma of five repeats) | the second term is about 5.38 ms (+0.3%); no routed design can meet it, so 1.05x is used as the working cap pending the user's decision |

## 6. Decisions for the user

1. **Lowering.** The conditional top-1 engine is the only routed design inside the cap, and it contradicts the
   signed wording "no conditional operators". It is top-1 only: top-2 weights depend on the image and cannot be
   pre-built into branches (a hard, equal-weight top-2 would need six branches).
2. **Residual experts.** Applied to all 39 layers they cost 8-13% over the like-for-like YOLO26-M. Scale-and-shift
   deltas on the 8 block outputs cost 3.9% over the like-for-like YOLO26-M (6.7% over the baseline engine), but
   that is feature modulation with 7,680 routed values per expert, not E-fold kernels.
3. **Latency cap.** 1.05x, or the voted min() rule, which nothing routed can pass.

## 7. Files

- Code (branch `dev/ds-yolo` of YOLO-Master, uncommitted): `ultralytics/nn/modules/moe/weight_bank.py`,
  `ultralytics/cfg/models/26/yolo26-wb.yaml`, the `weight_bank` hook in `ultralytics/nn/tasks.py`, the router
  weight-decay fix and `moe_router_decay_scale` in the trainer, `tests/test_weight_bank.py`.
- `ds-yolo-phase1-receipts.tar.gz`: batch logs (`batchA`-`batchE`, `micro`), per-layer profiles of every engine,
  the T4 scripts, the export scripts and their logs, and the Phase 0 CPU check logs.
- On the T4: `/root/ds/` (ONNX graphs, engines, dumps `dumpml_*`).
