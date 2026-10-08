# may: Mixture of Anything on YOLO

Routed real-time detectors built on the YOLO26 family: mixtures of experts (MoE), attention (MoA), tokens (MoT),
resolution (MoR) and depth (MoD), plus whatever else can be routed per image or per region inside one static
TensorRT engine. The research question is narrow and measurable: **can a routed YOLO beat the dense YOLO front at
its own average latency, on the device of record, by a margin a reviewer accepts?**

This repository starts empty on purpose. Code arrives only when a construct has passed its receipts (see
"Method"). Until then it carries the plan, the rules and the state of knowledge, so that every contributor and every
agent starts from the same facts.

## State of knowledge (2026-10-08)

Everything below was measured on the user's own hardware with one unchanged runtime bundle (TensorRT 10.16.1 fp16,
batch 1, static engines). Accuracy is multi-label pycocotools on full COCO val2017. Numbers for designs that have not
been trained are predictions and are labelled so.

**The fronts.** Dense YOLO26 on the NVIDIA T4: AP = 0.5261 + 0.0102 x (latency - 5.36 ms) around YOLO26-M. On the
NVIDIA L4 every engine runs at 0.47-0.56x its T4 time, the front is 1.9x steeper per millisecond, and no ranking
changes. YOLO26 dominates YOLO11, YOLOv12 and YOLOv13 at every scale on both devices; the attention families also pay
1.1-1.2 ms of CPU NMS outside the engine.

**What routing costs on the T4.** Any runtime tensor fed into a layer breaks the convolution-activation fusion
(+0.014-0.04 ms per layer); a convolution whose kernel is a runtime tensor costs about +0.08 ms per layer; one coarse
conditional (ONNX `If`) per image is free; bytes read per frame cost nothing measurable. Consequences, all measured:
per-image weight banks over the 39 1x1 convolutions miss the 1.05x cap in every static lowering (1.11-1.66x);
residual deltas 1.07-1.26x; soft per-pixel mixtures cost 7.5-9.5x a dense layer; token top-k with scatter does not
build in a static engine; a conditional engine with four static branches runs at 0.966x, but so does a dense model
inside the same `If`, so conditional routing is free and no faster.

**Where the headroom is, and is not.** From per-image detection dumps of the public YOLO26 scales: the per-image
oracle between M and L is large (0.545-0.558 at L shares of 0.16-0.42), but no ground-truth-free feature predicts
which image gains from more depth (rank correlation 0.00-0.03). The object count is the one predictable scene
variable (a 320-pixel thumbnail predicts it at Spearman 0.6-0.7); it carries +0.002-0.003 over a random route for
depth and +0.012-0.014 for input resolution (one weight set at 512 or 768). COCO has no scene an expert could own
(scene clusters explain 4-5% of class entropy). Tile-level routing is the one place a large share of the M-to-L gap
is reachable without ground truth: sending the 16% of 64-pixel tiles the model is least sure about to a stronger
model captures 0.85 of the gap on the dumps (0.5393 against a random-tile null of 0.5287), where every per-image
router managed +0.002.

**Training facts.** The released YOLO-Master routers are constants because the trainer decays router biases and
norms (issue and additive fix filed in the fork). Fine-tuning a converged public checkpoint for 10-20 epochs under
the fork's expressible YOLO26 recipe drops it to about 0.495-0.50 before it recovers, the same curve a run from the
Objects365 checkpoint reaches at epoch 19; four terms of YOLO26's recipe are not expressible in the fork. The unit
of evidence is therefore an 80-epoch run from the public Objects365 checkpoints with a dense control trained the
same way, never a short fine-tune of public weights.

**Running and pending.** An 80-epoch three-arm test (dense YOLO26-M; conditional top-1 with four branches;
conditional six pair branches) ends 2026-10-09; the routed arms trail the dense arm by 0.002-0.011 at equal epochs
and are a predicted negative. Seminar 5 (ten directions, three rounds) is in progress; its feasibility table will
be appended here.

## Method (the rules every row follows)

1. **Devices of record.** The T4 for designs that compete on latency (same MACs or fewer); the L4 for designs that
   buy accuracy with compute. Never mix devices inside one comparison; every number carries its device.
2. **The bar.** AP at least the dense front at the design's own average latency plus 0.003 (paired bootstrap
   2 sigma on val2017 is 0.0025), and at least the share-matched null plus 0.003 (a random route at the same share
   of images sent to the expensive branch). The room between dense M and dense L is never cited as headroom.
3. **The claim.** "+x AP over the share-matched null at the stated share, at an average latency of L ms on the T4
   (worst case W ms), in one static engine with one conditional"; the L4 column measured, never derived.
4. **Receipts before training.** Every construct buys, in order: a CPU headroom bound from the dumps (oracle /
   share null / ground-truth rule / learned router), a build-and-time receipt of the engine on the device of record
   with the comparator in the same session, then one 80-epoch run with a dense control and a kill rule written
   before launch. A miss is reported as a negative with every row; nothing is re-run without a named cause.
5. **Attribution.** A routed result must also beat its merged or permuted-route twin, so that the gain is routing
   and not capacity or recipe.

## Directions and status

| direction | construct on the table | status |
|---|---|---|
| MoE, weight space | per-image kernel banks, residual deltas, count-indexed weight soups behind one `If` | static lowerings closed by cost; conditional form is the running test (predicted negative) |
| MoR, resolution | one weight set at 512 / 768, thumbnail count router, one `If` | strongest per-image signal (+0.012 over null on public weights); needs an 80-epoch run trained for it |
| MoD, depth | M stem with M / L tails; level-allocated nested depth | no per-image signal for depth; count router +0.002-0.003; predicted negative |
| MoP, pyramid levels | per-image drop of the P3 path / add a P2 leaf ("level ladder") | signal strong (zero-small-object images at AUC 0.83); headroom thin (+0.002) |
| MoA, attention | zero-initialised attention block behind one `If` at P4 | gain is on large-object scenes, +0.002 over null; pairs naturally with MoR |
| MoT, tokens | tile-choice refiner: top-16 uncertain tiles to a window-attention expert, static TopK/Gather | best headroom on the dumps (0.85 of the M-to-L gap); expert efficiency unmeasured; L4 device |
| MoM, models | thumbnail router over N/S/M/L or heterogeneous families | four-width engine measured; same count signal as MoD |
| MoH, heads | NMS-free vs one-to-many + NMS per image | mechanism real (crowded objects) but 5% of objects; +0.001 over null |
| MoS, domains | scene-cluster experts branched late | label-space routing closed at zero GPU; no owned scene in COCO |
| MoC, compute | INT8 stem with INT8 or fp16 tail by saturation statistics | needs an INT8 front first; predicted negative |

## Roadmap

1. One T4 session: time L at 448-576 and M at 448-608, the thumbnail routers and the Resize-fed conditional engine, with
   dense M in the same session, to fix the envelope.
2. The two 4-hour frozen-trunk receipts on the H200: the tile expert (learned route against random tiles, k = 0 and all
   tiles from one checkpoint) and the rung specialist (routed-image head against an all-image head and a 640 control).
3. One 80-epoch run per surviving construct with its dense control, kill rules pre-registered, then the T4 and L4
   rows under the protocol.
4. The write-up ships under every outcome: the two-device method, the measured tables, the dense controls and the
   negatives; a routed chapter is added only when a row clears the bar.

## Related work in this programme

The routing modules, trainer fixes, receipt scripts and seminar records live in the YOLO-Master fork (branch
`dev/ds-yolo`) and in the project's deliverable archive until a construct earns its place here.
