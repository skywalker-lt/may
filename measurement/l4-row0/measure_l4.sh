#!/usr/bin/env bash
# L4 row 0: the same protocol as the T4 table (unchanged 1.2.0 trt10-cuda12 bundle, TensorRT 10.16.1 fp16, batch 1, 640,
# TensorRT-native median of 500 iterations, yolo26m re-timed every round). Engines are built beside each ONNX.
set -uo pipefail
B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; P=/root/venv/bin/python
IMG=/data/datasets/coco/images/val2017; D=/data/tmp/l4-row0; L=$D/l4.log; mkdir -p $D/out $D/profiles
P1=/data/tmp/ds-yolo/phase1/onnx; P1B=/data/tmp/ds-yolo/phase1b/onnx; X=$D/onnx
declare -A SRC
for m in yolo26m_ref wb_top2_conv wb_top2_matmul wb_top2_matmul_local wb_top1_matmul_local wb_top1_if if_pair6 res_scale res_bias res_lowrank res_scale_out wb_soft_conv wb_soft_matmul; do SRC[$m]=$P1/$m.onnx; done
for m in if_dense4 if_distinct4; do SRC[$m]=$P1B/$m.onnx; done
for f in $X/*.onnx; do m=$(basename $f .onnx); SRC[$m]=$f; done
ORDER="yolo26m_ref wb_top2_conv wb_top2_matmul wb_top2_matmul_local wb_top1_matmul_local wb_top1_if if_pair6 if_dense4 if_distinct4 res_scale res_bias res_lowrank res_scale_out wb_soft_conv wb_soft_matmul yolo26n yolo26s yolo26m yolo26l yolo26x yolo11n yolo11s yolo11m yolo11l yolo11x yolov12n_sdpa yolov12s_sdpa yolov12m_sdpa yolov12l_sdpa yolov12x_sdpa yolov13n_sdpa yolov13s_sdpa yolov13l_sdpa yolov13x_sdpa esmoe_m_sdpa esmoe_m yolov12m yolov13l"
log() { echo "$*" | tee -a $L; }
log "host: $(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader) | $($BIN --version 2>&1 | head -1) | $(date -u +%FT%TZ)"
for m in $ORDER; do
  f=${SRC[$m]:-}; [ -n "$f" ] && [ -f "$f" ] || { log "== $m: missing"; continue; }
  d=$(dirname $f); [ -f $d/$m.metadata.yaml ] || cp $X/yolo26m.metadata.yaml $d/$m.metadata.yaml
  if ! ls $d/$m-*-nvidia_l4-*.engine >/dev/null 2>&1 && ! ls $d/$m-*l4*-fp16.engine >/dev/null 2>&1; then
    t0=$(date +%s); timeout 3000 $BIN -m $f -s $IMG -b trt --precision fp16 --no-save --quiet --limit 5 > $D/out/build_$m.log 2>&1
    log "== build $m: exit=$? $(grep -E '\[trt\] built' $D/out/build_$m.log | sed -E 's/ -> .*\//  /' | cut -c1-110) ($(( $(date +%s) - t0 )) s)"
  fi
done
eng() { ls $(dirname ${SRC[$1]})/$1-*-fp16.engine 2>/dev/null | grep -i "l4" | head -1; }
BASE=$(eng yolo26m)
for r in 1 2 3 4 5; do
  log "-- round $r $(date -u +%T) temp $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)"
  log "$($P $D/trt_time.py $BASE 2>&1 | grep -E 'TRTTIME|rror')"
  for m in $ORDER; do e=$(eng $m); [ -n "$e" ] || continue; [ "$m" = yolo26m ] && continue
    log "$($P $D/trt_time.py $e 2>&1 | grep -E 'TRTTIME|rror' | sed "s/TRTTIME [^ ]*/TRTTIME $m/")"
  done
done
for m in $ORDER; do e=$(eng $m); [ -n "$e" ] || continue; log "$($P $D/trt_time.py $e --iters 100 --profile $D/profiles/profile_$m.txt 2>&1 | grep PROFILE | sed "s/^/$m /")"; done
# accuracy sanity row (AP is expected device-independent): yolo26m dump on the L4
source /root/venv/bin/activate; rm -rf $D/dumpml_yolo26m
timeout 3600 $BIN -m $X/yolo26m.onnx -s $IMG -b trt --precision fp16 --conf 0.001 --iou 0.7 --multi-label --save-txt $D/dumpml_yolo26m --no-save --quiet 2>&1 | grep -E "^\[summary\]" | sed "s/^/  yolo26m dumpml: /" | tee -a $L
$P $D/txt_to_coco_eval.py $D/dumpml_yolo26m --ann /data/datasets/coco/annotations/instances_val2017.json 2>&1 | grep -E "PYCOCO" | tee -a $L
log "DONE $(date -u +%T)"
