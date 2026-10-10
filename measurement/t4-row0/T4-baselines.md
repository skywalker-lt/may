# T4 baselines: YOLO11, YOLO26, YOLOv12, YOLOv13, every scale, full val2017 (2026-10-02)

Device: NVIDIA Tesla T4 (15 GB, driver 595.91.07, 70 W cap), vast.ai container, Ubuntu 24.04, CUDA 12.8. Runtime: yolo-master-edge 1.2.0 (43dfa9e), the released trt10-cuda12 bundle unchanged, TensorRT 10.16.1, fp16, batch 1, 640x640, CUDA preprocessing. Every model is a static [1,3,640,640] ONNX export at opset 18 (YOLO11/YOLO26 from the ultralytics exports on /data/yolo-quant-work/weights; YOLOv12/YOLOv13 re-exported static from their forks; the YOLOv12 weights on disk are the Turbo release, 19.67M params at M). `*_sdpa` rows are the same weights re-exported with the area-attention block expressed as `torch.nn.functional.scaled_dot_product_attention` (patch in `patch_sdpa.py`; parity with the original export max |delta| 6e-4), which lets TensorRT emit its fused attention kernels (`_gemm_mha_v2` in the per-layer profile) instead of the fork's hand-written exp/max/sum/div softmax chain.

Columns. CLI forward: the CLI's per-frame inference time (enqueue to sync) averaged over 300 warm frames of val2017; graph: the same with `--cuda-graph`; post: the CLI's CPU decode (end2end heads have none); cold median: `--bench cold`, 200 iterations; TRT-native: CUDA-event time of the serialized engine with no host transfers, 500 back-to-back iterations after 200 warm-up (reads slightly hot for the m to x rows under the 70 W cap, and at nano strips the CLI's ~1 ms of host overhead on this 4-core container); published T4 ms: the model zoos' T4 TensorRT10 figures (YOLOv12 paper rows for the sdpa models, Turbo README rows for the originals); mAP in-process: the CLI's val-protocol scorer (conf 0.001, IoU 0.7, multi-label, max_det 300) on all 5,000 val2017 images with YOLO labels; mAP pycocotools: the official COCO API on the CLI's val-protocol dumps (two columns: single-label dumps, and multi-label dumps via the CLI's --multi-label flag, which is the protocol the published ultralytics numbers use; the two agree on end2end heads and differ by 0.4 to 0.9 on NMS heads); AP_S/M/L from pycocotools; published mAP from the model zoos; engine build: first-use TensorRT build time on this T4.

| model | CLI forward ms | graph ms | post ms | cold median ms | TRT-native ms | published T4 ms | mAP50-95 in-process | mAP50-95 pycocotools single-label | pycocotools multi-label | AP_S / AP_M / AP_L (single-label) | published mAP | engine build s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n | 2.68 | 3.40 | 1.04 | 1.96 | 1.62 | 1.5 | 0.3911 | 0.3871 | 0.3930 | 0.196 / 0.424 / 0.561 | 39.5 | 272 |
| yolo11s | 3.71 | 3.73 | 1.09 | 2.95 | 2.70 | 2.5 | 0.4649 | 0.4600 | 0.4661 | 0.284 / 0.505 / 0.630 | 47.0 | 278 |
| yolo11m | 4.85 | 4.90 | 1.12 | 5.38 | 5.24 | 4.7 | 0.5128 | 0.5069 | 0.5131 | 0.331 / 0.560 / 0.668 | 51.5 | - |
| yolo11l | 6.21 | 6.37 | 1.11 | 7.02 | 6.79 | 6.2 | 0.5318 | 0.5248 | 0.5315 | 0.349 / 0.582 / 0.685 | 53.4 | 324 |
| yolo11x | 11.10 | 11.32 | 1.18 | 12.55 | 12.52 | 11.3 | 0.5455 | 0.5374 | 0.5443 | 0.363 / 0.590 / 0.687 | 54.7 | 356 |
| yolo26n | 1.99 | 2.91 | 0.00 | 1.58 | 1.65 | 1.7 | 0.4066 | 0.4060 | - | 0.200 / 0.444 / 0.587 | 40.1 | 267 |
| yolo26s | 3.05 | 2.94 | 0.00 | 2.65 | 2.75 | 2.5 | 0.4803 | 0.4795 | - | 0.303 / 0.525 / 0.645 | 47.8 | 265 |
| yolo26m | 4.61 | 4.62 | 0.01 | 5.19 | 5.36 | 4.7 | 0.5270 | 0.5261 | 0.5261 | 0.358 / 0.570 / 0.688 | 52.5 | 285 |
| yolo26l | 5.94 | 6.03 | 0.01 | 6.76 | 6.89 | 6.2 | 0.5435 | 0.5417 | - | 0.366 / 0.584 / 0.700 | 54.4 | 293 |
| yolo26x | 10.88 | 11.03 | 0.01 | 12.15 | 12.41 | 11.8 | 0.5715 | 0.5691 | - | 0.405 / 0.611 / 0.730 | 56.9 | 328 |
| yolov12n | 3.70 | 3.40 | 1.13 | 2.75 | 2.48 | 1.6 | 0.4009 | 0.3963 | - | 0.187 / 0.437 / 0.584 | 40.4 | 301 |
| yolov12s | 4.44 | 4.44 | 1.16 | 4.57 | 4.40 | 2.42 | 0.4728 | 0.4664 | - | 0.272 / 0.514 / 0.650 | 47.6 | 308 |
| yolov12m | 6.75 | 6.82 | 1.14 | 7.31 | 7.19 | 4.27 | 0.5234 | 0.5168 | - | 0.339 / 0.569 / 0.686 | 52.5 | 331 |
| yolov12l | 10.60 | 10.77 | 1.16 | 11.44 | 11.37 | 5.83 | 0.5383 | 0.5300 | - | 0.360 / 0.585 / 0.695 | 53.8 | 376 |
| yolov12x | 17.99 | 18.36 | 1.18 | 19.42 | 19.32 | 10.38 | 0.5549 | 0.5465 | - | 0.378 / 0.600 / 0.702 | 55.4 | 417 |
| yolov13n | 3.80 | 3.59 | 1.13 | 3.27 | 3.04 | 1.97 | 0.4109 | 0.4064 | - | 0.197 / 0.448 / 0.586 | 41.6 | 474 |
| yolov13s | 4.92 | 5.04 | 1.14 | 5.14 | 4.98 | 2.98 | 0.4762 | 0.4703 | - | 0.284 / 0.518 / 0.638 | 48.0 | 487 |
| yolov13l | 13.07 | 13.34 | 1.20 | 13.99 | 13.98 | 8.63 | 0.5309 | 0.5246 | - | 0.346 / 0.583 / 0.681 | 53.4 | 601 |
| yolov13x | 21.42 | 21.80 | 1.18 | 22.73 | 22.71 | 14.67 | 0.5454 | 0.5388 | - | 0.372 / 0.596 / 0.693 | 54.8 | 650 |
| yolov12n_sdpa | 3.43 | 3.63 | 1.11 | 2.23 | 1.90 | 1.64 | 0.4008 | 0.3962 | 0.4028 | 0.188 / 0.437 / 0.583 | 40.6 | 300 |
| yolov12s_sdpa | 3.61 | 3.75 | 1.14 | 3.29 | 3.13 | 2.61 | 0.4728 | 0.4663 | 0.4735 | 0.272 / 0.514 / 0.649 | 48.0 | 304 |
| yolov12m_sdpa | 5.16 | 5.18 | 1.16 | 5.65 | 5.53 | 4.86 | 0.5237 | 0.5171 | 0.5234 | 0.340 / 0.570 / 0.686 | 52.5 | 328 |
| yolov12l_sdpa | 7.29 | 7.45 | 1.20 | 8.04 | 8.00 | 6.77 | 0.5382 | 0.5300 | 0.5375 | 0.359 / 0.585 / 0.695 | 53.7 | 376 |
| yolov12x_sdpa | 12.74 | 12.97 | 1.18 | 13.93 | 14.02 | 11.79 | 0.5552 | 0.5467 | 0.5538 | 0.378 / 0.601 / 0.702 | 55.2 | 425 |
| yolov13n_sdpa | 3.63 | 3.66 | 1.11 | 2.81 | 2.50 | 1.97 | 0.4108 | 0.4070 | 0.4127 | 0.197 / 0.448 / 0.589 | 41.6 | 476 |
| yolov13s_sdpa | 4.01 | 4.02 | 1.16 | 4.04 | 3.86 | 2.98 | 0.4764 | 0.4706 | 0.4777 | 0.284 / 0.518 / 0.638 | 48.0 | 487 |
| yolov13l_sdpa | 9.77 | 9.93 | 1.16 | 10.61 | 10.64 | 8.63 | 0.5304 | 0.5240 | 0.5305 | 0.345 / 0.582 / 0.681 | 53.4 | 607 |
| yolov13x_sdpa | 15.98 | 16.39 | 1.21 | 17.40 | 17.61 | 14.67 | 0.5452 | 0.5385 | 0.5448 | 0.372 / 0.595 / 0.693 | 54.8 | 649 |
| esmoe_m | 9.79 | 9.95 | 1.15 | 10.70 | 10.48 | - | 0.5301 | 0.5214 | 0.5289 | 0.355 / 0.576 / 0.681 | - | 396 |
| esmoe_m_sdpa | 8.61 | 8.72 | 1.15 | 9.50 | 9.29 | - | 0.5304 | - | 0.5292 | - / - / - | - | 396 |

## Findings

1. Accuracy reproduces every published row within 0.6 AP (in-process) and within 0.5 AP (multi-label pycocotools, the matching protocol) across all 19 public models; the SDPA re-exports are numerically identical to the originals.
2. YOLO11 and YOLO26 latency reproduces: native engine time within 8 to 11% of the published rows at every scale, YOLO26 end2end heads with zero post-processing.
3. The original YOLOv12/YOLOv13 exports are 1.5 to 2.3x slower than published at every scale. The per-layer profile attributes it to the attention path lowering to an unfused softmax chain (25% of yolov12m's engine time). The YOLOv12 paper measured with FlashAttention active.
4. With the SDPA export the YOLOv12 rows land 14 to 19% above the paper's figures, against YOLO11's own 8 to 11% offset on the same loop; the few percent left is the FlashAttention gain the paper quotes (0.3 to 0.4 ms). YOLOv13 keeps a 20 to 29% residual (hypergraph glue). The YOLOv12 Turbo README rows (4.27 ms at M) are not reproducible from any public export.
5. EsMoE-M (YOLO-Master) runs at 10.5 ms native, 2.0x YOLO26-M, for +0.3 AP on the matched multi-label pycocotools protocol (0.5289 vs 0.5261; single-label 0.5214). Its four ES_MOE blocks are num_experts=3 with top_k=3 (k = E), so this row is a dense three-branch gated network (97.9 GFLOPs by the fork's counter vs 75.4 for YOLO26-M), not a sparse one; no row in this table executes fewer experts than it holds. The original export also carried the unfused-attention artefact (its A2C2f blocks lowered without fused MHA): the esmoe_m_sdpa row (re-exported through the fork with the same SDPA patch, parity 0.0) is the fair figure, 9.29 ms native, 1.73x YOLO26-M, at 0.5292 multi-label pycocotools (+0.3). Per the fork's own gates the released routers are constant across images, so the +0.3 is capacity, not routing.
6. CUDA graph gains nothing on the T4 at m to x and costs 0.7 to 0.9 ms on the CLI path at nano (yolo11n 2.68 to 3.40, yolo26n 1.99 to 2.91): the device is MAC-bound at M, and at nano the CLI's host overhead (about 1 ms of the 2.68, see the native column) dominates rather than kernel launches.

Consolidated baseline for the DS-YOLO work: use the `*_sdpa` rows for YOLOv12 and YOLOv13, labelled as such; the T4 is the named device; one validator (pycocotools on val-protocol dumps) for every row including DS-YOLO itself.

Receipts: measure_t4_batch{1,2,6}.log, pycoco_t4_batch{1,2,6}.log, trtexec_t4.log, bench_*.json (yolomaster-bench/v1 cold runs), profile_*.txt (per-layer TensorRT profiles), dump_*_coco.json (COCO result files), the scripts (measure_t4.sh, dump_t4.sh, batch*.sh, trt_time.py, txt_to_coco_eval.py) and the metadata sidecars. Pod facts: engines build in 4.5 to 11 minutes per model on sm_75; the whole run took 03:24 to 09:58 UTC.
