## 4. First receipt (under 4 H200 hours, no training) and its kill rule

1. Dump the full (d = 1..6, K = 100/300, scale 512/640) grid of public RT-DETRv2-R50 and -S on all 5,000 val2017 images
   from one backbone-encoder pass per image (the probe script, GPU; about 20 min), score every point with multi-label
   pycocotools, and fit the convergence router out-of-fold on the 300-query layer-3 signals; mix on the CPU. About 1 H200 h.
2. Build the DX-3/6 graph (one `If`, identity vs layers 4-6) with TensorRT 10.16.1 on the H200 for parity (within 0.0005
   of the CPU mixture) and the build check; profile the decoder's per-layer share. About 1 H200 h.
3. One T4 session line (the session the programme already plans): dense R50 at 640, Det3, Det6, DX-3/6 on real inputs,
   beside dense YOLO26-M. This fixes the 0.45 ms per layer and the branch penalty on the device of record.
Kill rules, written now: (a) learned route minus the share-matched 3/6 null under +0.003 at share 0.5 on full val2017;
(b) routed AP under dense Det5 + 0.003 at the routed point's est. latency (the lens's own envelope); (c) the router's
Spearman with the ground-truth count above 0.6 and no residual correlation with the gain (a count code in disguise).
Any one of (a)-(c) closes the line. (d) stands regardless: the row is 0.02 under the YOLO26 envelope, so it can only be
published as a DETR-family result, which the user must want.

## 5. Training plan and cost (only after the receipt)

Exit-aware fine-tune: backbone and encoder frozen, decoder layers and heads 3 and 6 plus a two-layer router MLP on the
pooled layer-3 queries trained 12 epochs on COCO under RT-DETRv2's public recipe (which reproduces 0.534 on this volume,
so the fork's 0.004 recipe shortfall does not apply), head 3 trained with the routed images up-weighted and the router
by a straight-through gate with the balance term, biases and norms out of decay (`ISSUE_router_weight_decay.md`). Est.
6-8 H200 h (R50's 72-epoch public run is about 70 H200-h equivalent; 12 epochs with two thirds of the network frozen).
A dense control at the same epochs (Det6 heads fine-tuned the same way) is the same cost again. Gate: routed minus the
control's own Det5 point at equal est. latency >= +0.003, and >= +0.003 over the share null.

## 6. Prior art

RT-DETR's "flexible speed tuning by adjusting the number of decoder layers" (Lv et al., 2023, Table 5) and RF-DETR's
droppable queries and prunable decoder layers (2025) are dense knobs chosen per deployment. AnyDepth-DETR/-YOLO
(arXiv 2605.09407, May 2026) trains one network whose backbone and neck depth is set at inference without retraining
(up to 1.82x speed-up for 2.0 AP), again per deployment as far as its abstract states. DEED (NAACL 2024) and DeeCap
(CVPR 2022) exit a decoder per step or per image with confidence, for text generation and captioning. AnytimeYOLO (2025)
places early exits in YOLO. DynamicDet, MSDNet, SkipNet and Mixture-of-Depths are the per-image depth family the
seminars cited. I found no published per-image, confidence- or convergence-gated decoder exit for a DETR detector; the
novelty is the per-image convergence signal in query space and the static one-`If` lowering with measured T4 cost. It is
a modest novelty, and it sits in the DETR literature, not the programme's.
