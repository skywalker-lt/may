#!/usr/bin/env bash
# Batch 6: all nine YOLOv12/YOLOv13 scales re-exported with scaled_dot_product_attention (supersedes batch 4). Waits for batches 2-3 and the upload marker.
set -uo pipefail
while pgrep -f "^bash /root/(measure_t4|dump_t4|batch2_t4|batch3_diag).sh$" >/dev/null; do sleep 60; done
while [ ! -f /root/xfer/batch6_ready ]; do sleep 30; done
M6="yolov12n_sdpa yolov12s_sdpa yolov12m_sdpa yolov12l_sdpa yolov12x_sdpa yolov13n_sdpa yolov13s_sdpa yolov13l_sdpa yolov13x_sdpa"
MODELS="$M6" bash /root/measure_t4.sh; cp /root/measure_t4.log /root/measure_t4_batch6.log
source /venv/main/bin/activate >/dev/null 2>&1; export LD_LIBRARY_PATH=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0/lib
for m in $M6; do e=$(ls /root/xfer/$m-*-tesla_t4-trt10.16.1-fp16.engine 2>/dev/null | head -1); [ -n "$e" ] || continue; python /root/trt_time.py $e 2>&1 | grep TRTTIME | tee -a /root/trtexec_t4.log; done
for m in yolov12m_sdpa yolov13l_sdpa yolov12n_sdpa; do e=$(ls /root/xfer/$m-*-tesla_t4-trt10.16.1-fp16.engine 2>/dev/null | head -1); [ -n "$e" ] || continue; python /root/trt_time.py $e --iters 100 --profile /root/profile_$m.txt 2>&1 | grep PROFILE | tee -a /root/trtexec_t4.log; done
sed -i 's/^while pgrep.*$/true/' /root/dump_t4.sh; MODELS="$M6" bash /root/dump_t4.sh; cp /root/pycoco_t4.log /root/pycoco_t4_batch6.log
echo "BATCH6 DONE $(date +%H:%M:%S)" | tee -a /root/trtexec_t4.log
# then the batch-1 dumps (formerly batch 5)
MODELS="yolo11m yolo26m yolov12m yolo11l yolo26l yolov12l yolov13l esmoe_m" bash /root/dump_t4.sh; cp /root/pycoco_t4.log /root/pycoco_t4_batch1.log
echo "BATCH5 DONE $(date +%H:%M:%S)" | tee -a /root/pycoco_t4_batch1.log /root/trtexec_t4.log
