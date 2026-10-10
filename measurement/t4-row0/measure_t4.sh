#!/usr/bin/env bash
# Row 0 of the DS-YOLO plan: the published-table comparators through the v1.2.0 CLI on a T4.
# TensorRT 10.16 fp16, batch 1, 640; warm 200-frame timing, bench cold median, graph on/off; full val2017 accuracy.
set -uo pipefail
B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; X=/root/xfer
IMG=/root/coco/images/val2017; L=/root/measure_t4.log
MODELS=${MODELS:-"yolo11m yolo26m yolov12m yolo11l yolo26l yolov12l yolov13l esmoe_m"}
ACC=${ACC:-1}
: > $L
log() { echo "$*" | tee -a $L; }
log "host: $(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader) | $($BIN --version 2>&1 | head -1) | $(date -u +%FT%TZ)"
for m in $MODELS; do
  M=$X/$m.onnx; [ -f $M ] || { log "== $m: missing"; continue; }
  pa="-b trt --precision fp16"
  t0=$(date +%s)
  first="$(timeout 3600 $BIN -m $M -s $IMG $pa --no-save --quiet --limit 5 2>&1 | grep -E "\[trt\] built|\[model\]" | sed -E 's/\[model\] [^ ]+ +//' | tr '\n' ' ' | cut -c1-240)"
  log "== $m fp16: $first (engine step $(( $(date +%s) - t0 )) s)"
  for extra in "" "--cuda-graph"; do
    t="$(timeout 900 $BIN -m $M -s $IMG $pa $extra --no-save --quiet --warmup 20 --limit 300 2>&1 | grep -E '^\[summary\]' | sed -E 's/.*avg\/frame: //; s/ +wall=.*//')"
    b="$(timeout 900 $BIN -m $M -s $IMG $pa $extra --bench cold --bench-iters 200 --bench-warmup 20 --bench-json /root/bench_${m}_fp16${extra:+_graph}.json --no-save --quiet 2>&1 | grep -oE 'median=[0-9.]+ms p90=[0-9.]+' | head -1)"
    log "  $m fp16 ${extra:-plain}: $t | bench $b"
  done
  if [ "$ACC" = 1 ]; then
    a="$(timeout 7200 $BIN -m $M -s $IMG $pa --cuda-graph --accuracy /root/coco/labels/val2017 --save-txt /root/txt_$m --no-save --quiet 2>&1 | grep -E '^\[accuracy\]' | sed -E 's/\[accuracy\] +//')"
    log "  $m fp16 accuracy val2017: $a"
  fi
done
log "MEASURE DONE $(date +%H:%M:%S)"
