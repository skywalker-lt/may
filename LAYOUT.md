# Repository layout (scripts and configs only; no weights, engines or detection dumps)

| path | what it holds |
|---|---|
| `fork/` | the routed-model code from the YOLO-Master fork's `dev/ds-yolo` branch: `ultralytics/nn/modules/moe/weight_bank.py` (per-image weight-bank routing, fixed routers, conditional lowering), the model YAMLs `yolo26-wb.yaml` and `yolo26l-mstem.yaml`, `tests/test_weight_bank.py`, and `scripts/ds_yolo/` (80-epoch launcher and trainer, upcycling, disk cache, monitor, splice builder, receipt launcher and evaluation, gate mixing, dump batches, final scoring, the stage-2 recipe YAML). These files depend on the fork and run from inside it. |
| `runs80/` | the training configuration (`args.yaml`) of the three 80-epoch arms and their README with the final numbers |
| `receipts/phase0`, `phase1`, `phase1b` | Phase 0 CPU checks and Phase 1 T4 engine receipts: ONNX exporters, conditional (`If`) builders, micro-benchmarks, residual prototypes, T4 batch scripts, ONNX sidecar YAMLs |
| `receipts/phase2` | seminar-4 engines: two-scale, M/L two-tail, four-width and splice builders, per-input profiling, L4 micro graphs |
| `receipts/receipts_conditional_controls*` | dense-in-`If`, pair and distinct conditional controls, route-agreement scripts |
| `measurement/t4-row0` | T4 baseline protocol: measurement and dump scripts, pycocotools scorer, model sidecars, the baseline table |
| `measurement/l4-row0` | L4 session scripts, engine-build helper, timing tools, sidecars |
| `measurement/t4-row1` | the 2026-10-09 T4 envelope session script and sidecars |
| `seminars/seminar3`-`seminar6/agent-scripts` | every script the seminar agents wrote to compute headroom, nulls, envelopes and costs from the dumps (CPU, pycocotools) |

Data that the scripts read (per-image val2017 dumps, pooled stem features, weights) is not included; the results they produced
are summarised in `SUMMARY.md` and archived with the project's experiment workbook.
