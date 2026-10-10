#!/usr/bin/env bash
# After measure_t4.sh: val-protocol dumps (conf 0.001, iou 0.7) for every model, scored with pycocotools against instances_val2017.json.
set -uo pipefail
B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; X=/root/xfer; IMG=/root/coco/images/val2017; L=/root/pycoco_t4.log
true
source /venv/main/bin/activate >/dev/null 2>&1
: > $L
MODELS=${MODELS:-"yolo11m yolo26m yolov12m yolo11l yolo26l yolov12l yolov13l esmoe_m"}; for m in $MODELS; do
  M=$X/$m.onnx; [ -f $M ] || continue
  rm -rf /root/dump_$m
  timeout 3600 $BIN -m $M -s $IMG -b trt --precision fp16 --cuda-graph --conf 0.001 --iou 0.7 --save-txt /root/dump_$m --no-save --quiet 2>&1 | grep -E '^\[summary\]' | sed "s/^/  $m dump: /" | tee -a $L
  python /root/txt_to_coco_eval.py /root/dump_$m 2>&1 | grep -E "images_with|PYCOCO" | tee -a $L
done
echo "PYCOCO DONE $(date +%H:%M:%S)" | tee -a $L
