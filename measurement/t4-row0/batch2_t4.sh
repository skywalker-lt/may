#!/usr/bin/env bash
# Batch 2: the remaining 12 scales. Waits for batch 1 (timing + pycocotools) and for the upload marker, then runs the same two passes.
set -uo pipefail
while pgrep -f measure_t4.sh >/dev/null || pgrep -f dump_t4.sh >/dev/null; do sleep 60; done
while [ ! -f /root/xfer/batch2_ready ]; do sleep 30; done
cp /root/measure_t4.log /root/measure_t4_batch1.log; cp /root/pycoco_t4.log /root/pycoco_t4_batch1.log
M2="yolo11n yolo11s yolo11x yolo26n yolo26s yolo26x yolov12n yolov12s yolov12x yolov13n yolov13s yolov13x"
MODELS="$M2" bash /root/measure_t4.sh
cp /root/measure_t4.log /root/measure_t4_batch2.log
sed -i 's/^while pgrep.*$/true/' /root/dump_t4.sh
MODELS="$M2" bash /root/dump_t4.sh
cp /root/pycoco_t4.log /root/pycoco_t4_batch2.log
echo "BATCH2 DONE $(date +%H:%M:%S)" >> /root/measure_t4_batch2.log
