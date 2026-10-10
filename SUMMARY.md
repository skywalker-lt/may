# may: final summary (abandoned 2026-10-10)

`may` (Mixture of Anything on YOLO) set out to find a routed real-time detector that beats the dense YOLO26 front on COCO
at equal latency on the NVIDIA T4. After six seminars, about thirty routed constructs, one 80-epoch three-arm training test
and every receipt the plan called for, no construct came close enough to fund. COCO is abandoned as the target; the work
moves to dataset-specific benchmarks (AI-TOD-v2, VisDrone), where the objects are small and the arithmetic below differs.

## The result in one paragraph

On YOLO26-class detectors at M to L size and 640-pixel input, on a T4-class GPU, measured as COCO AP at equal latency against
the best dense model at any input scale, no form of dynamic sparsity we could build or bound beats dense. Per-image routing is
limited by a signal that carries only the object count, which dense compound scaling already exploits. Per-location routing
has a strong signal but no cheap expert that knows more than the base model's own features. The T4 makes any sparse execution
dearer than a fused dense convolution.

## The decisive measurements

**The 80-epoch test** (YOLO26-M upcycled from the Objects365 checkpoint, the fork's YOLO26 stage-2 recipe, batch 64, seed 0;
validator AP on full val2017, last / best weights):

| arm | AP | vs dense |
|---|---|---|
| dense YOLO26-M | 0.5090 / 0.5094 | |
| six-pair routed (four experts per routed 1x1 conv, top-2 hard pairs, fixed linear router) | 0.5072 / 0.5076 | -0.0018 |
| top-1 routed (four experts, top-1, fixed linear router) | 0.5002 / 0.5009 | -0.0085 |

The rule set before the run (a routed arm at least +0.003 over dense) was not met. Forced-branch dumps at epoch 20 show the
experts specialise by about +0.003 on their own images and lose 0.009-0.015 to the data split.

**The measured dense envelope on the T4** (TensorRT 10.16 fp16, batch 1, multi-label pycocotools):

| model at input | AP | T4 ms |
|---|---|---|
| YOLO26-M at 640 | 0.5261 | 5.34 |
| YOLO26-L at 512 | 0.5262 | 4.88 |
| YOLO26-L at 544 | 0.5329 | 5.32 |
| YOLO26-L at 576 | 0.5368 | 6.25 |
| YOLO26-L at 640 | 0.5417 | 6.89 |

L at 544 beats M at 640 by +0.0068 AP at the same latency. Every routed margin found in the programme was smaller than what
this envelope absorbs. Two further dense facts raise it again: native-aspect inference with padding (est. +0.005 to +0.009 at
4.6-6.0 ms) and public LW-DETR-L (+0.019 over YOLO26-L at 640 on full val2017).

**The T4 cost model** (Phase 1 and the envelope session): any runtime tensor into a layer breaks the convolution-activation
fusion (+0.014-0.04 ms per layer); a runtime-kernel convolution costs about +0.08 ms per layer; scatter-based token routing does
not build in a static engine; a per-image conditional is free but no faster (a dense model inside the same conditional runs
at the same speed); every branch of a two-scale conditional pays +0.25-0.5 ms on real inputs.

## What was closed, by measurement

| construct | why it closed |
|---|---|
| static per-image weight banks on the 39 pointwise convs | every lowering 1.11-1.66x dense latency |
| residual expert deltas | 1.07-1.26x |
| same-architecture per-image experts behind a conditional | the 80-epoch test above |
| per-image depth (tail selection, block skipping) | YOLO26 has no removable depth: public L with units bypassed scores 0.002 (C3k2 concatenates its units) |
| pyramid-level routing | the P4/P5-only branch scores 0.18 untrained; misroutes cost more than the level saves |
| per-image resolution routing (512/640, 448/608, two-scale) | absorbed by dense L at reduced and native-aspect input; fails even under ground-truth routing at 512 |
| attention on/off routing | YOLO26's attention is load-bearing (ablated: 0.351); routing-specific share of an added block 0.06-0.08 |
| one-to-one vs one-to-many head routing | the per-image head gap is noise; the one-to-many head is a constant +0.006 |
| scene / domain experts on COCO | scene specialisation 0.000 +- 0.001; scene clusters are a noisy count code |
| INT8 precision routing | INT8 loss 0.022; saturation tracks nothing; Q/DQ inside a conditional does not build |
| soft per-pixel mixtures, token top-k, crop re-resolution, cascades | cost: 7.5-9.5x a dense layer, unbuildable, or the second pass never pays |
| mixture of public models (exchange routing) | every gain decomposes into resolution, instance weighting and compound scaling |
| budget routing | a scheduler over the dense points absorbs it |

## What still had signal (and why it was not enough on COCO)

- **Per-tile and per-query routing.** The model's own uncertainty concentrates 0.85-0.91 of a stronger model's gain into 16% of
  the tiles (random tiles: 0.12-0.20). The static TopK/Gather graph builds on TensorRT. But 75-91% of that gain is box geometry
  that same-resolution features do not contain, and with a larger teacher the expert needs a fidelity no datum supports.
- **Routes whose variable is not content**: stem statistics chosen by a measured noise level (robustness, +0.072 on severity-3
  noise, -0.227 if misapplied to clean images), the previous frame's own detections in video, a content rectangle on a
  MAC-bound phone CPU. Each changes the claim or the device rather than winning COCO AP on the T4.

## Why the new targets differ

On small-object benchmarks the per-location signal measured here is the relevant one: objects cover a small fraction of the
frame, resolution (not capacity) carries most of the gain, and a router that finds where the objects are can spend
high-resolution compute there. The COCO-specific obstacles (a count-only per-image signal, a dense envelope that already
trades capacity for pixels) are weaker in that regime. That is a hypothesis carried over, not a result.

## Where the material is

Seminar syntheses and transcripts, receipts and the experiment workbook (all parsed tables plus every raw log) are in the
project's deliverable archive; training code, weight-bank modules and receipt scripts are on the YOLO-Master fork's
`dev/ds-yolo` branch; the 80-epoch weights, configurations and logs are archived with the experiment data.
