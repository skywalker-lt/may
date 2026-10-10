#!/usr/bin/env bash
# Batch 4: the SDPA re-exports of yolov12m and yolov13l (attention as scaled_dot_product_attention instead of a hand-written softmax),
# same two passes as the comparators, plus TRT-native timing and a per-layer profile. Tests whether the v12/v13 latency gap is the export.
set -uo pipefail
while pgrep -f "^bash /root/(measure_t4|dump_t4|batch2_t4|batch3_diag).sh$" >/dev/null; do sleep 60; done
M4="yolov12m_sdpa yolov13l_sdpa"
MODELS="$M4" bash /root/measure_t4.sh; cp /root/measure_t4.log /root/measure_t4_batch4.log
sed -i 's/^while pgrep.*$/true/' /root/dump_t4.sh; MODELS="$M4" bash /root/dump_t4.sh; cp /root/pycoco_t4.log /root/pycoco_t4_batch4.log
source /venv/main/bin/activate >/dev/null 2>&1; export LD_LIBRARY_PATH=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0/lib
for m in $M4; do e=$(ls /root/xfer/$m-*-tesla_t4-trt10.16.1-fp16.engine 2>/dev/null | head -1); [ -n "$e" ] || continue
  python /root/trt_time.py $e 2>&1 | grep TRTTIME | tee -a /root/trtexec_t4.log
  python /root/trt_time.py $e --iters 100 --profile /root/profile_$m.txt 2>&1 | grep PROFILE | tee -a /root/trtexec_t4.log; done
echo "BATCH4 DONE $(date +%H:%M:%S)" | tee -a /root/trtexec_t4.log
