# Agent 10, round 1: lens 10 (a different base)

## 0. The lens against the files, in numbers

The programme's comparator is the measured YOLO26 (model, scale) envelope on the T4. Before any route, a change of base
must survive that envelope at its own latency. It does not, for every attention-family base on file or on disk
(`work/agent10/envelope_gaps.log`; envelope interpolated between the measured points L448/L512/L544/L640/X640; published
RT-DETR rows converted from FPS and also shown with this loop's +8-11% measured-over-published offset, est.):

| base at 640 | T4 ms | AP | YOLO26 envelope at that ms | gap |
|---|---|---|---|---|
| YOLOv12-M SDPA (measured) | 5.53 | 0.5234 | 0.5341 | -0.011 |
| YOLOv12-L SDPA (measured) | 8.00 | 0.5375 | 0.5472 | -0.010 |
| YOLOv13-L SDPA / EsMoE-M SDPA (measured) | 10.64 / 9.29 | 0.5305 / 0.5292 | 0.5603 / 0.5536 | -0.030 / -0.024 |
| RT-DETRv2-S (R18), published 217 FPS | 4.61 (5.1 est.) | 0.481 | 0.520 (0.529) | -0.039 (-0.048) |
| RT-DETRv2-M (R50-m), published 145 FPS | 6.90 (7.6 est.) | 0.519 | 0.542 (0.545) | -0.023 (-0.026) |
| RT-DETRv2-L (R50), published 108 FPS; 0.534 re-measured on this volume | 9.26 (10.2 est.) | 0.534 | 0.554 (0.558) | -0.020 (-0.024) |
| RT-DETRv2-X (R101), published 74 FPS | 13.5 | 0.543 | 0.569 | -0.026 |

Two consequences fix what this lens can and cannot deliver. First, the largest routing margin the programme has ever
measured over a null is +0.003 (per-image) or +0.0106 (per-tile, below its own ceiling); the smallest base-change
deficit is 0.010 (YOLOv12-M). No route recovers a base deficit of 0.010-0.048, so **no construct on another base can
produce a programme row (envelope + 0.003 on the T4)**; the same holds on the L4, where every engine runs at 0.47-0.56 of
its T4 time and the front is 1.9x steeper. Second, "token routing is native" in YOLOv12 buys nothing on the T4: in the
yolov12m_sdpa per-layer profile the fused attention kernels total 0.32 ms of 5.74 ms (5.6%), so even a route that
removed attention entirely saves 0.3 ms, worth 0.0015 AP at the envelope's slope above 5.3 ms (0.004-0.005 AP per ms).
Routing tokens inside YOLOv12's area attention is closed by arithmetic before it is built.

What survives the files is narrower than the lens: the DETR family has one routable object that YOLO26 structurally
lacks and that the seminars closed only *because* YOLO26 lacks it. Per-image depth was closed on YOLO26 since "nothing can
be dropped untrained" (bypassed L scores 0.002). An RT-DETR decoder is trained with a box and score head on every layer
(deep supervision), so its depth is removable without training: the paper's own table (R50, T4 TensorRT fp16, published)
gives 6/5/4/3/2/1 decoder layers = 53.1/53.0/52.7/52.4/51.3/49.1 AP at 9.3/8.8/8.3/7.9/7.5/7.0 ms, about 0.45 ms per
layer (a quarter of the model's time for 2% of its FLOPs: the decoder is launch-bound, the one place on the T4 where
skipped work is skipped time). The family also has a native query count K (the encoder's static TopK/Gather of 300 tokens,
exactly the lowering the packet says builds). That is the construct below. It is proposed with its honest standing: a
row against the **RT-DETR family's own dense (depth, K, scale) envelope**, the comparator this lens defines, and a
labelled negative against the YOLO26 envelope.

## 1. Construct: DX-3/6, a per-image decoder-exit route on RT-DETRv2

**What routes.** RT-DETRv2-R50 (public 6x weights, 0.534 on this volume) runs backbone, hybrid encoder, the static
top-300 query selection and decoder layers 1-3 for every image. After layer 3 a router reads the query set itself and
decides, once per image, whether the image is finished (emit head 3's boxes and scores) or runs layers 4-6 (emit head 6).
**On what signal.** Not the object count. The decoder is an iterative refiner, so it exposes a signal YOLO26 cannot
produce: the convergence of the query set, measured as the mean IoU between the top-30 boxes at layer 2 and at layer 3,
plus the top-30 score mass at layer 3. An image whose boxes stopped moving by layer 3 gains nothing from three more
refinement steps; an image whose boxes are still moving does. The packet's finding that every per-image signal is a
count code is tested directly below (Spearman of the signal with the ground-truth count, and with the per-image gain).
**Lowering.** One `If` after layer 3: branch A is an identity on (boxes_3, logits_3); branch B is layers 4-6 and head 6
(the branches have identical output shapes, 300 x 4 and 300 x 80). The existing postprocess TopK(300)/Gather follows.
No scatter, no runtime kernel, batch 1, the same lowering as the programme's conditional engines. Worst case = dense
R50 plus the branch penalty.
**Cost on the device of record (T4, est. from the paper's per-layer cost and `T4-envelope.md`'s branch penalty).**
Branch A 7.9 ms, branch B 9.3 ms, +0.25-0.5 ms for the conditional (0.3 taken): average 8.2 + 1.35 s ms at full share s,
i.e. 8.9 ms at s = 0.5, 8.5 ms at s = 0.25; worst case 9.6 ms (1.03x dense R50). The programme's measured T4 loop runs
8-11% above published rows, so every figure here is est. until an RT-DETR engine is timed in the same session as dense M.
The penalty is 0.67 of a decoder layer: the route must skip at least one layer on every image on average just to pay for
its own `If`, which is why the exit point is layer 3 and not 5.

## 2. Why the comparators cannot absorb it

**The lens's comparator (RT-DETRv2's own dense envelope over decoder depth, K and scale)** is concave in depth: removing
layers 6, 5, 4 costs 0.1, 0.3, 0.3 AP and layers 3, 2 cost 1.1, 2.2. A random 3/6 route at share 0.5 therefore realises
the chord (0.5275 at 8.6 ms, published AP) and sits 0.0025 under dense Det5 (0.530 at 8.8) before the branch penalty; with
the penalty the routed point at 8.9 ms must beat Det5 + 0.003 = 0.533, within 0.001 of the full model while skipping half
the decoder on half the images. That is only possible if the per-image gain of layers 4-6 is concentrated on a
recognisable half of the images; the section below measures whether it is. **The device-of-record comparator** asks
0.5535 + 0.003 at 8.9 ms (YOLO26-L/X interpolation); the oracle of this route cannot exceed dense R50 by more than the
per-image oracle gain measured below, so the construct is 0.02 under the YOLO26 bar by construction and is reported as a
negative there.

