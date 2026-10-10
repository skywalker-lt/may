# 80-epoch three-arm test (DS-YOLO), fetched from the H200 on 2026-10-10

Fork branch `dev/ds-yolo` (scripts/ds_yolo/: train_80ep.py, launch80.sh, upcycle_fixed.py, final_eval80.py).
Recipe: YOLO26-M stage 2 as the fork expresses it (`YOLO-Master/scripts/ds_yolo/recipe_yolo26m_stage2.yaml`), batch 64,
80 epochs from the Objects365 checkpoint yolo26m-objv1-150.pt, cache=disk, seed 0.

| run | model | final AP (validator, full val2017; last / best) |
|---|---|---|
| dense80 | yolo26m.yaml + yolo26m-objv1-150.pt | 0.5090 / 0.5094 |
| pair680 | weights/yolo26m-objv1-wb-pair6-fixed.pt (four experts, top-2 hard pairs, fixed linear router) | 0.5072 / 0.5076 |
| top180 | weights/yolo26m-objv1-wb-top1-fixed.pt (four experts, top-1, fixed linear router) | 0.5002 / 0.5009 |

Per run: `args.yaml` (full config), `results.csv` (per-epoch metrics), `weights/{best,last,last_healthy}.pt`
(last_healthy carries the optimiser state). Raw logs: `<run>.out` (trainer stdout), `train.log` (one line per run per
minute), `final_eval.log` and `final_eval/` (the closing validation; its own pycocotools recompute is invalid because
predictions.json uses 0-79 category ids, use the validator numbers). Checksums of every .pt verified against the pod.
