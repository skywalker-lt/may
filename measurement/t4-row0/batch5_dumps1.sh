#!/usr/bin/env bash
# Batch 5: the val-protocol dumps + pycocotools for the batch-1 models (skipped by the quoting bug), after batch 4.
set -uo pipefail
while pgrep -f "^bash /root/(measure_t4|dump_t4|batch2_t4|batch3_diag|batch4_sdpa).sh$" >/dev/null; do sleep 60; done
sed -i 's/^while pgrep.*$/true/' /root/dump_t4.sh
MODELS="yolo11m yolo26m yolov12m yolo11l yolo26l yolov12l yolov13l esmoe_m" bash /root/dump_t4.sh; cp /root/pycoco_t4.log /root/pycoco_t4_batch1.log
echo "BATCH5 DONE $(date +%H:%M:%S)" >> /root/pycoco_t4_batch1.log
