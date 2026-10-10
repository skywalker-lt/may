#!/usr/bin/env bash
while pgrep -f "queueB.sh|^bash ./ds_phase1.sh" > /dev/null; do sleep 20; done
./ds_phase1.sh "res_scale res_bias res_lowrank res_full" 5 batchC > out/batchC.out 2>&1
P=/venv/main/bin/python; L=out/micro.log; : > $L
for m in micro_1x1_static micro_1x1_bank micro_3x3_static micro_3x3_bank; do $P onnx/micro_t4.py onnx/$m.onnx onnx/$m.engine >> out/micro_build.log 2>&1; done
for r in 1 2 3; do for m in micro_1x1_static micro_1x1_bank micro_3x3_static micro_3x3_bank; do $P /root/trt_time.py onnx/$m.engine 2>&1 | grep -E "TRTTIME|rror" | sed "s/TRTTIME [^ ]*/TRTTIME $m/" >> $L; done; done
for m in micro_1x1_static micro_1x1_bank micro_3x3_static micro_3x3_bank; do $P /root/trt_time.py onnx/$m.engine --iters 100 --profile out/profile_$m.txt 2>&1 | grep PROFILE | sed "s/^/$m /" >> $L; done
echo "DONE $(date -u +%T)" >> $L
