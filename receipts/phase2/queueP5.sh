#!/usr/bin/env bash
cd /root/ds; P=/venv/main/bin/python; B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; L=out/batchP5.log; : > $L
cd onnxP2; sed "s/^- 640$/- 768/" /root/xfer/yolo26m.metadata.yaml > if_res2_force0.metadata.yaml; cd ..
t0=$(date +%s); timeout 3000 $BIN -m onnxP2/if_res2_force0.onnx -s /root/coco/images/val2017 -b trt --precision fp16 --no-save --quiet --limit 5 > out/build_if_res2_force0.log 2>&1
echo "== build if_res2_force0: exit=$? ($(( $(date +%s) - t0 )) s)" >> $L
E=$(ls onnxP2/if_res2_force0-*-fp16.engine | head -1); M512=$(ls onnxP2/yolo26m_512-*-fp16.engine | head -1); M768=$(ls onnxP2/yolo26m_768-*-fp16.engine | head -1); R2=$(ls onnxP2/if_res2-*-fp16.engine | head -1)
IN="onnxP2/in768_0.npy onnxP2/in768_1.npy onnxP2/in768_2.npy onnxP2/in768_3.npy"
for r in 1 2 3; do $P onnxF/trt_time_in.py $E --inputs $IN --tag if_res2_branch512 2>&1 | grep TRTIN >> $L; $P onnxF/trt_time_in.py $R2 --inputs $IN --tag if_res2_branch768 2>&1 | grep TRTIN >> $L; $P /root/trt_time.py $M512 2>&1 | grep TRTTIME | sed "s/TRTTIME [^ ]*/TRTTIME yolo26m_512/" >> $L; done
$P onnxP2/trt_prof_in.py $E $IN 2>&1 | grep -E "PROFIN|rror" >> $L
echo "DONE $(date -u +%T)" >> $L
