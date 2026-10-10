# Agent 6 request: forced-branch dumps of the running weight-bank arms (seminar 5, after round 2)

## What to run (H200, no training; est. 0.7 H200 h for the top-1 arm and the dense arm, +0.7 h for the optional pair arm)

The script `make_forced_branches.py` (this folder) turns a weight-bank checkpoint into plain YOLO26-M checkpoints, one per
forced branch: every `BankConv2d` (39 banked 1x1 convs in layers 6-22) is replaced by an `nn.Conv2d` holding one expert's
kernel (or the average of a hard pair, or of all four), so the result has no weight-bank module and goes through
`scripts/ds_yolo/dump_batch.sh` like any dense `.pt`. It also writes the arm's own routing on val2017 (image_id -> expert
indices) so the routed AP is computed exactly on the CPU by per-image mixing of the forced dumps.

```bash
cd /data/YOLO-Master; export PYTHONPATH=/data/YOLO-Master
S=<path of this folder>/make_forced_branches.py; O=/data/weights/forced; V=/data/datasets/coco/images/val2017
# 1. top-1 arm, epoch-20 backup: four forced experts + the merged average + its own routing (about 5 min on the GPU)
python $S --ckpt /training_data/runs80/top180/weights/last.pt --tag top180_ep20 --out $O --merged --route-dir $V --device 0
# 2. optional, lower priority: the six-pair arm (6 pair branches; the 4 single experts come for free)
python $S --ckpt /training_data/runs80/pair680/weights/last.pt --tag pair680_ep20 --out $O --pairs --route-dir $V --device 0
# 3. dump spec = the generated lines + the dense arm at the same epoch
cat $O/spec_top180_ep20.txt > /tmp/spec_a6.txt
echo "dense80_ep20 /training_data/runs80/dense80/weights/last.pt 640 o2o" >> /tmp/spec_a6.txt
bash scripts/ds_yolo/dump_batch.sh /tmp/spec_a6.txt seminar5_agent6
```

Please check that the three backups are at the same epoch (`BACKUP_INFO` in each run folder) and copy to `dumps/`:
the six (or twelve) COCO json files, `route_top180_ep20.json` (and `route_pair680_ep20.json`), and `check_*.log`.
Each `check_*.log` must end in `PASS`; it compares, on a random input and two val images, the banked model with its gates
forced to each branch against the plain forced model (exact in fp32), checks that the natural top-1 route equals the
forced branch of the chosen expert, and reloads one saved file through `YOLO()` after the fp16 save.

## Smoke test (CPU, one thread, done 2026-10-08)

`OMP_NUM_THREADS=1 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python make_forced_branches.py --ckpt
/data/tmp/ds-yolo/weights/yolo26m-objv1-wb-init.pt --tag init --out smoke --perturb 0.05 --test-top1 --pairs --merged
--route-dir .../val2017 --route-limit 8 --batch 4` -> `smoke/check_init.log`: 39 banks, 4 experts; all 11 branches
(4 experts, 6 pairs, merged) match the banked model with forced gates exactly (rel. max err 0.0 on the head inputs);
expert 0 and expert 1 differ (3.46 on the head inputs, so the check is not vacuous); natural top-1 route equals the
forced branch; reload after fp16 save 2.8e-3; each file 45.2 MB; ONNX export of a forced file through the same
`YOLO(pt).export(format="onnx", ...)` call as `dump_batch.sh`: see `onnx_check.log`. The init checkpoint's experts are
identical copies (eps 0), so `--perturb` and `--test-top1` exist only for this test; never pass them on the pod.

## What I compute from the dumps (CPU, `work/agent6/mixacc.py`, exact per-image mixing)

1. The 4 x 4 specialisation matrix: AP of forced expert b on the images the router sends to expert c. Specialisation
   means the diagonal beats the column's other entries.
2. Routed AP (forced dumps mixed by `route_*.json`) against: the best single forced expert, the merged average, the
   dense arm at the same epoch, and the branch-permuted share null (5 draws).
3. Whether the experts' per-image differences line up with object count (Spearman with GT and with thumbnail count).

## Kill rule written now

If the mean diagonal advantage of the 4 x 4 matrix is under +0.002 AP and the routed AP is not above the best single
forced expert by +0.002, the same-architecture per-image experts show no specialisation at epoch 20: D's predicted
negative stands, and the late-branched count specialists of my round-2 construct lose their only measured support.

## Round-3 fix (2026-10-08 20:30 UTC)

The pod run failed at `model.to("0")` (`RuntimeError: Invalid device string: '0'`, `inputs/dumps_agent6/forced_a6.log`):
my script passed the `--device` string straight to torch. Fixed in place (a bare digit now maps to `cuda:N`; the round-2
version is kept as `make_forced_branches.r2.py`). Nothing else changed; the same commands re-run as written. Only the
dense arm's epoch-20 dump landed (0.4962).
