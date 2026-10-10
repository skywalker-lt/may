#!/usr/bin/env bash
# L4 micro graphs: also run on the T4 later for the cross-device per-layer comparison.
cd /data/tmp/l4-row0; while pgrep -f "measure_l4.sh|measure_l4_p2.sh" > /dev/null; do sleep 30; done
P=/root/venv/bin/python; D=/data/tmp/l4-row0; L=$D/l4_micro.log; : > $L; P2=/data/tmp/ds-yolo/phase2/onnx
for m in dense_p3 softmoe4_p3 dense_p4 softmoe4_p4 tokentopk_p3 attn_p4 dense_p4_2; do $P $D/micro_build.py $P2/l4micro_$m.onnx $D/out/l4micro_$m.engine >> $D/out/micro_build.log 2>&1 || echo "BUILD FAILED $m" >> $L; done
for r in 1 2 3; do for m in dense_p3 softmoe4_p3 dense_p4 softmoe4_p4 tokentopk_p3 attn_p4 dense_p4_2; do [ -f $D/out/l4micro_$m.engine ] && $P $D/trt_time.py $D/out/l4micro_$m.engine 2>&1 | grep -E "TRTTIME|rror" | sed "s/TRTTIME [^ ]*/TRTTIME $m/" >> $L; done; done
for m in dense_p3 softmoe4_p3 dense_p4 softmoe4_p4 tokentopk_p3 attn_p4; do [ -f $D/out/l4micro_$m.engine ] && $P $D/trt_time.py $D/out/l4micro_$m.engine --iters 100 --profile $D/profiles/profile_l4micro_$m.txt 2>&1 | grep PROFILE | sed "s/^/$m /" >> $L; done
echo "DONE $(date -u +%T)" >> $L
