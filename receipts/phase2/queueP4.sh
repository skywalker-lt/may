#!/usr/bin/env bash
cd /root/ds; P=/venv/main/bin/python; B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; L=out/batchP4.log; : > $L
BASE=/root/xfer/yolo26m-568fcaa87c34-tesla_t4-trt10.16.1-fp16.engine
cd onnxP2; sed "s/^- 640$/- 768/" /root/xfer/yolo26m.metadata.yaml > if_res2.metadata.yaml; cp /root/xfer/yolo26m.metadata.yaml splice_ml.metadata.yaml; cd ..
for m in if_res2 splice_ml; do t0=$(date +%s); timeout 3000 $BIN -m onnxP2/$m.onnx -s /root/coco/images/val2017 -b trt --precision fp16 --no-save --quiet --limit 5 > out/build_$m.log 2>&1
  echo "== build $m: exit=$? $(grep -E '\[trt\] built' out/build_$m.log | sed -E 's/ -> .*\//  /' | cut -c1-110) ($(( $(date +%s) - t0 )) s)" >> $L; done
R2=$(ls onnxP2/if_res2-*-fp16.engine | head -1); SP=$(ls onnxP2/splice_ml-*-fp16.engine | head -1); M768=$(ls onnxP2/yolo26m_768-*-fp16.engine | head -1); M512=$(ls onnxP2/yolo26m_512-*-fp16.engine | head -1); L640=$(ls /root/xfer/yolo26l-*-fp16.engine | head -1)
for r in 1 2 3 4 5; do for pair in "yolo26m $BASE" "splice_ml $SP" "yolo26l $L640" "if_res2 $R2" "yolo26m_512 $M512" "yolo26m_768 $M768"; do set -- $pair; $P /root/trt_time.py $2 2>&1 | grep TRTTIME | sed "s/TRTTIME [^ ]*/TRTTIME $1/" >> $L; done; done
IN="onnxP2/in768_0.npy onnxP2/in768_1.npy onnxP2/in768_2.npy onnxP2/in768_3.npy"
for r in 1 2 3; do $P onnxF/trt_time_in.py $R2 --inputs $IN --tag if_res2_real 2>&1 | grep TRTIN >> $L; $P onnxF/trt_time_in.py $M768 --inputs $IN --tag yolo26m_768_real 2>&1 | grep TRTIN >> $L; done
$P onnxP2/trt_prof_in.py $R2 $IN 2>&1 | grep -E "PROFIN|rror" >> $L
IN6="onnxF/in_branch0.npy onnxF/in_branch1.npy onnxF/in_branch2.npy onnxF/in_branch3.npy"
for r in 1 2; do $P onnxF/trt_time_in.py $SP --inputs $IN6 --tag splice_ml_real 2>&1 | grep TRTIN >> $L; $P onnxF/trt_time_in.py $L640 --inputs $IN6 --tag yolo26l_real 2>&1 | grep TRTIN >> $L; done
ls -la onnxP2/if_res2-*.engine onnxP2/splice_ml-*.engine | awk '{print $5, $9}' >> $L
echo "DONE $(date -u +%T)" >> $L
