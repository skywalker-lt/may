#!/usr/bin/env bash
# DS-YOLO Phase 1 on the T4: build engines with the unchanged 1.2.0 bundle, then time them interleaved with yolo26m.
# usage: ds_phase1.sh "<model names>" [rounds]   (ONNX in /root/ds/onnx; engines are cached beside them)
set -uo pipefail
B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; D=/root/ds; P=/venv/main/bin/python
BASE=/root/xfer/yolo26m-568fcaa87c34-tesla_t4-trt10.16.1-fp16.engine
MODELS=${1:?models}; ROUNDS=${2:-5}; TAG=${3:-run}; L=$D/out/${TAG}.log
log() { echo "$*" | tee -a $L; }
log "host: $(nvidia-smi --query-gpu=name,driver_version,temperature.gpu --format=csv,noheader) | $($BIN --version 2>&1 | head -1) | $(date -u +%FT%TZ)"
for m in $MODELS; do
  dir=$D/onnx; [ "$m" = yolo26m_ref_rebuild ] && { dir=$D/rebuild; mm=yolo26m_ref; } || mm=$m
  if ! ls $dir/$mm-*-fp16.engine >/dev/null 2>&1; then
    t0=$(date +%s)
    timeout 3000 $BIN -m $dir/$mm.onnx -s /root/coco/images/val2017 -b trt --precision fp16 --no-save --quiet --limit 5 > $D/out/build_$m.log 2>&1
    log "== build $m: exit=$? $(grep -E '\[trt\] built' $D/out/build_$m.log | sed -E 's/ -> .*\//  /' | cut -c1-120) ($(( $(date +%s) - t0 )) s) $(grep -ciE 'error' $D/out/build_$m.log) error lines"
  fi
done
for r in $(seq 1 $ROUNDS); do
  log "-- round $r $(date -u +%T) temp $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)"
  log "$($P /root/trt_time.py $BASE 2>&1 | grep -E 'TRTTIME|rror')"
  for m in $MODELS; do
    dir=$D/onnx; [ "$m" = yolo26m_ref_rebuild ] && { dir=$D/rebuild; mm=yolo26m_ref; } || mm=$m
    e=$(ls $dir/$mm-*-fp16.engine 2>/dev/null | head -1); [ -n "$e" ] || { log "TRTTIME $m MISSING"; continue; }
    log "$($P /root/trt_time.py $e 2>&1 | grep -E 'TRTTIME|rror' | sed "s/TRTTIME [^ ]*/TRTTIME $m/")"
  done
done
for m in $MODELS; do
  dir=$D/onnx; [ "$m" = yolo26m_ref_rebuild ] && { dir=$D/rebuild; mm=yolo26m_ref; } || mm=$m
  e=$(ls $dir/$mm-*-fp16.engine 2>/dev/null | head -1); [ -n "$e" ] || continue
  log "$($P /root/trt_time.py $e --iters 100 --profile $D/out/profile_$m.txt 2>&1 | grep PROFILE | sed "s/^/$m /")"
done
$P /root/trt_time.py $BASE --iters 100 --profile $D/out/profile_yolo26m_base.txt 2>&1 | grep PROFILE | sed "s/^/yolo26m_base /" | tee -a $L
log "DONE $(date -u +%T)"
