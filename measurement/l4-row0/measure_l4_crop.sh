#!/usr/bin/env bash
cd /data/tmp/l4-row0; while pgrep -f "measure_l4" > /dev/null; do sleep 30; done
P=/root/venv/bin/python; D=/data/tmp/l4-row0; L=$D/l4_crop.log; : > $L; P2=/data/tmp/ds-yolo/phase2/onnx
for m in yolo26m_b4_320 yolo26m_b4_160; do $P $D/micro_build.py $P2/$m.onnx $D/out/$m.engine >> $D/out/micro_build.log 2>&1 || echo "BUILD FAILED $m" >> $L; done
BASE=$(ls $D/onnx/yolo26m-*l4*-fp16.engine | head -1)
for r in 1 2 3; do $P $D/trt_time.py $BASE 2>&1 | grep TRTTIME >> $L; for m in yolo26m_b4_320 yolo26m_b4_160; do [ -f $D/out/$m.engine ] && $P $D/trt_time.py $D/out/$m.engine 2>&1 | grep TRTTIME | sed "s/TRTTIME [^ ]*/TRTTIME $m/" >> $L; done; done
echo "DONE $(date -u +%T)" >> $L
