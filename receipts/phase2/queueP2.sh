#!/usr/bin/env bash
cd /root/ds; P=/venv/main/bin/python; B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; IMG=/root/coco/images/val2017; L=out/batchP2.log; : > $L
BASE=/root/xfer/yolo26m-568fcaa87c34-tesla_t4-trt10.16.1-fp16.engine
source /venv/main/bin/activate >/dev/null 2>&1
cd onnxP2; for m in yolo26m_512:512 yolo26m_768:768 yolo26s_768:768 if_depth_ml:640; do n=${m%%:*}; sz=${m##*:}; sed "s/^- 640$/- $sz/" /root/xfer/yolo26m.metadata.yaml > $n.metadata.yaml; done; cd ..
for m in yolo26m_512 yolo26m_768 yolo26s_768 if_depth_ml; do
  t0=$(date +%s); timeout 3000 $BIN -m onnxP2/$m.onnx -s $IMG -b trt --precision fp16 --no-save --quiet --limit 5 > out/build_$m.log 2>&1
  echo "== build $m: exit=$? $(grep -E '\[trt\] built' out/build_$m.log | sed -E 's/ -> .*\//  /' | cut -c1-110) ($(( $(date +%s) - t0 )) s)" >> $L
done
for m in stem_m_640 stem_n_640 stem_n_320; do $P onnx/micro_t4.py onnxP2/$m.onnx onnxP2/$m.engine >> out/build_stems.log 2>&1; done
for r in 1 2 3 4 5; do
  $P /root/trt_time.py $BASE 2>&1 | grep TRTTIME >> $L
  for m in yolo26m_512 yolo26m_768 yolo26s_768 if_depth_ml; do e=$(ls onnxP2/$m-*-fp16.engine 2>/dev/null | head -1); [ -n "$e" ] && $P /root/trt_time.py $e 2>&1 | grep TRTTIME | sed "s/TRTTIME [^ ]*/TRTTIME $m/" >> $L; done
  for m in stem_m_640 stem_n_640 stem_n_320; do $P /root/trt_time.py onnxP2/$m.engine 2>&1 | grep TRTTIME | sed "s/TRTTIME [^ ]*/TRTTIME $m/" >> $L; done
done
IFE=$(ls onnxP2/if_depth_ml-*-fp16.engine | head -1); IN="onnxF/in_branch0.npy onnxF/in_branch1.npy onnxF/in_branch2.npy onnxF/in_branch3.npy"
for r in 1 2 3; do $P onnxF/trt_time_in.py $IFE --inputs $IN --tag if_depth_ml_real 2>&1 | grep TRTIN >> $L; done
$P onnxP2/trt_prof_in.py $IFE $IN 2>&1 | grep -E "PROFIN|rror" >> $L
for m in yolo26m_512 yolo26m_768 yolo26s_768; do
  rm -rf dumpml_$m; timeout 3600 $BIN -m onnxP2/$m.onnx -s $IMG -b trt --precision fp16 --conf 0.001 --iou 0.7 --multi-label --save-txt /root/ds/dumpml_$m --no-save --quiet 2>&1 | grep -E "^\[summary\]" | sed "s/^/  $m dumpml: /" >> $L
  python /root/txt_to_coco_eval.py /root/ds/dumpml_$m 2>&1 | grep -E "PYCOCO" | sed "s/PYCOCO dumpml_/PYCOCOML /" >> $L
done
echo "DONE $(date -u +%T)" >> $L
