#!/usr/bin/env bash
cd /root/ds; while pgrep -f "queueP2.sh" > /dev/null; do sleep 20; done
P=/venv/main/bin/python; B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; L=out/batchP3.log; : > $L
BASE=/root/xfer/yolo26m-568fcaa87c34-tesla_t4-trt10.16.1-fp16.engine
cp /root/xfer/yolo26m.metadata.yaml onnxP2/if_scale4.metadata.yaml
t0=$(date +%s); timeout 3000 $BIN -m onnxP2/if_scale4.onnx -s /root/coco/images/val2017 -b trt --precision fp16 --no-save --quiet --limit 5 > out/build_if_scale4.log 2>&1
echo "== build if_scale4: exit=$? $(grep -E '\[trt\] built' out/build_if_scale4.log | sed -E 's/ -> .*\//  /' | cut -c1-110) ($(( $(date +%s) - t0 )) s)" >> $L
E=$(ls onnxP2/if_scale4-*-fp16.engine | head -1); IN="onnxF/in_branch0.npy onnxF/in_branch1.npy onnxF/in_branch2.npy onnxF/in_branch3.npy"
for r in 1 2 3; do $P /root/trt_time.py $BASE 2>&1 | grep TRTTIME >> $L; $P onnxF/trt_time_in.py $E --inputs $IN --tag if_scale4_real 2>&1 | grep TRTIN >> $L; done
$P onnxP2/trt_prof_in.py $E $IN 2>&1 | grep -E "PROFIN|rror" >> $L
ls -la onnxP2/*.engine | awk '{print $5, $9}' >> $L
echo "DONE $(date -u +%T)" >> $L
