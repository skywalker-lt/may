#!/usr/bin/env bash
# Batch 3 (after batches 1-2): TensorRT-native timing of every cached engine (CUDA events, no transfers) plus per-layer profiles
# of the attention-family engines, to separate CLI overhead from engine time and to see what area attention / hypergraph cost on Turing.
set -uo pipefail
while pgrep -f batch2_t4.sh >/dev/null || pgrep -f measure_t4.sh >/dev/null || pgrep -f dump_t4.sh >/dev/null; do sleep 60; done
source /venv/main/bin/activate >/dev/null 2>&1
B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; L=/root/trtexec_t4.log; : > $L
export LD_LIBRARY_PATH=$B/lib:${LD_LIBRARY_PATH:-}
for e in /root/xfer/*-tesla_t4-trt10.16.1-fp16.engine; do python /root/trt_time.py $e 2>&1 | grep -E "TRTTIME|Error|error" | tee -a $L; done
for m in yolov12m yolov13l yolo11m yolo26m yolov12n yolov13n; do
  e=$(ls /root/xfer/$m-*-tesla_t4-trt10.16.1-fp16.engine 2>/dev/null | head -1); [ -n "$e" ] || continue
  python /root/trt_time.py $e --iters 100 --profile /root/profile_$m.txt 2>&1 | grep -E "PROFILE|Error|error" | tee -a $L
done
echo "DIAG DONE $(date +%H:%M:%S)" | tee -a $L
