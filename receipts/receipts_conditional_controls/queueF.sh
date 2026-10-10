#!/usr/bin/env bash
# Seminar 3 round-1 requests: conditional-engine controls, rebuild variance, per-branch timing with real inputs, route agreement.
cd /root/ds; P=/venv/main/bin/python; B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge
BASE=/root/xfer/yolo26m-568fcaa87c34-tesla_t4-trt10.16.1-fp16.engine; L=out/batchF2.log; : > $L
./ds_phase1.sh "if_dense4 if_dense2 if_distinct4 wb_top1_if" 5 batchF > out/batchF.out 2>&1
t0=$(date +%s); timeout 3000 $BIN -m rebuild2/wb_top1_if.onnx -s /root/coco/images/val2017 -b trt --precision fp16 --no-save --quiet --limit 5 > out/build_wb_top1_if_rebuild.log 2>&1
echo "== build wb_top1_if_rebuild: exit=$? ($(( $(date +%s) - t0 )) s) $(ls -la rebuild2/*.engine | awk "{print \$5}") bytes" >> $L
RB=$(ls rebuild2/wb_top1_if-*-fp16.engine | head -1); IF=$(ls onnx/wb_top1_if-*-fp16.engine | head -1); DI=$(ls onnx/if_distinct4-*-fp16.engine | head -1); DE=$(ls onnx/if_dense4-*-fp16.engine | head -1)
for r in 1 2 3 4 5; do
  $P /root/trt_time.py $BASE 2>&1 | grep TRTTIME >> $L
  $P /root/trt_time.py $RB 2>&1 | grep TRTTIME | sed "s/TRTTIME [^ ]*/TRTTIME wb_top1_if_rebuild/" >> $L
done
IN="onnxF/in_branch0.npy onnxF/in_branch1.npy onnxF/in_branch2.npy onnxF/in_branch3.npy"
for r in 1 2 3; do
  echo "-- real-input round $r" >> $L
  $P onnxF/trt_time_in.py $BASE --inputs $IN --tag yolo26m_base 2>&1 | grep TRTIN >> $L
  $P onnxF/trt_time_in.py $IF --inputs $IN --tag wb_top1_if_alternating 2>&1 | grep TRTIN >> $L
  for i in 0 1 2 3; do $P onnxF/trt_time_in.py $IF --inputs onnxF/in_branch$i.npy --tag wb_top1_if_only_branch$i 2>&1 | grep TRTIN >> $L; done
  $P onnxF/trt_time_in.py $DI --inputs $IN --tag if_distinct4_alternating 2>&1 | grep TRTIN >> $L
  $P onnxF/trt_time_in.py $DE --inputs $IN --tag if_dense4_alternating 2>&1 | grep TRTIN >> $L
done
$P /root/trt_time.py $IF --iters 100 --profile out/profile_wb_top1_if_again.txt 2>&1 | grep PROFILE >> $L
$P onnxF/route_agree.py onnxF/wb_top1_if_dbg.onnx onnxF/router_only.onnx 2>&1 | grep -E "ROUTE|rror|Traceback" >> $L
ls -la onnx/if_*.engine onnx/wb_top1_if-*.engine rebuild2/*.engine | awk "{print \$5, \$9}" >> $L
echo "DONE $(date -u +%T)" >> $L
