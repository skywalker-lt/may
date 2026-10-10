#!/usr/bin/env bash
cd /data/tmp/l4-row0; while pgrep -f "measure_l4" > /dev/null; do sleep 30; done
P=/root/venv/bin/python; D=/data/tmp/l4-row0; L=$D/l4_micro2.log; : > $L; P2=/data/tmp/ds-yolo/phase2/onnx
$P $D/micro_build.py $P2/l4micro_tokentopk_p3_k50.onnx $D/out/l4micro_tokentopk_p3_k50.engine >> $D/out/micro_build.log 2>&1 || echo "BUILD FAILED tokentopk_p3_k50" >> $L
for r in 1 2 3; do for m in dense_p3 tokentopk_p3 tokentopk_p3_k50; do [ -f $D/out/l4micro_$m.engine ] && $P $D/trt_time.py $D/out/l4micro_$m.engine 2>&1 | grep -E "TRTTIME|rror" | sed "s/TRTTIME [^ ]*/TRTTIME $m/" >> $L; done; done
echo "DONE $(date -u +%T)" >> $L
