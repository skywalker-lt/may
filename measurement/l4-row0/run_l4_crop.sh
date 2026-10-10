#!/usr/bin/env bash
# L4 crop-pass receipt (seminar 4, local re-resolution kill rule): batch-4 crops at 320 and 160 through YOLO26-M,
# same bundle/TensorRT as the L4 table; yolo26m re-timed in every round. Log: /data/tmp/l4-row0/l4_crop.log
P=/root/venv/bin/python; D=/data/tmp/l4-row0; L=$D/l4_crop.log; P2=/data/tmp/ds-yolo/phase2/onnx; mkdir -p $D/out
echo "host: $(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader) | $(date -u +%FT%TZ)" > $L
for m in yolo26m_b4_320 yolo26m_b4_160; do
  if [ ! -f $D/out/$m.engine ]; then t0=$(date +%s); $P $D/micro_build.py $P2/$m.onnx $D/out/$m.engine >> $D/out/micro_build_crop.log 2>&1 && echo "built $m ($(( $(date +%s) - t0 )) s)" >> $L || echo "BUILD FAILED $m" >> $L; fi
done
BASE=$(ls $D/onnx/yolo26m-*nvidia_l4*-fp16.engine | head -1)
for r in 1 2 3 4 5; do
  echo "-- round $r $(date -u +%T) temp $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)" >> $L
  $P $D/trt_time.py $BASE 2>&1 | grep -E "TRTTIME|rror" | sed "s/TRTTIME [^ ]*/TRTTIME yolo26m/" >> $L
  for m in yolo26m_b4_320 yolo26m_b4_160; do [ -f $D/out/$m.engine ] && $P $D/trt_time.py $D/out/$m.engine 2>&1 | grep -E "TRTTIME|rror" | sed "s/TRTTIME [^ ]*/TRTTIME $m/" >> $L; done
done
for m in yolo26m_b4_320 yolo26m_b4_160; do [ -f $D/out/$m.engine ] && $P $D/trt_time.py $D/out/$m.engine --iters 100 --profile $D/profiles/profile_l4crop_$m.txt 2>&1 | grep PROFILE | sed "s/^/$m /" >> $L; done
echo "DONE $(date -u +%T)" >> $L
