#!/usr/bin/env bash
cd /root/ds; while pgrep -f "queueP5.sh" > /dev/null; do sleep 20; done
P=/venv/main/bin/python; L=out/batchP6.log; : > $L
BASE=/root/xfer/yolo26m-568fcaa87c34-tesla_t4-trt10.16.1-fp16.engine; IF=$(ls onnxP2/if_depth_ml-*-fp16.engine | head -1); SP=$(ls onnxP2/splice_ml-*-fp16.engine | head -1); L640=$(ls /root/xfer/yolo26l-*-fp16.engine | head -1); S4=$(ls onnxP2/if_scale4-*-fp16.engine | head -1); REF=$(ls onnx/yolo26m_ref-*-fp16.engine | head -1)
IN="onnxF/in_branch0.npy onnxF/in_branch1.npy onnxF/in_branch2.npy onnxF/in_branch3.npy"
for r in 1 2 3; do for pair in "yolo26m_base $BASE" "yolo26m_ref $REF" "if_depth_ml $IF" "splice_ml $SP" "yolo26l $L640" "if_scale4 $S4"; do set -- $pair; $P onnxF/trt_time_in.py $2 --inputs $IN --tag $1 2>&1 | grep TRTIN >> $L; done; done
echo "DONE $(date -u +%T)" >> $L
