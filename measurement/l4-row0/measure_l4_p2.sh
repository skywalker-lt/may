#!/usr/bin/env bash
# L4 follow-up: resolution rows and the heterogeneous engines, timed with yolo26m in the same rounds.
cd /data/tmp/l4-row0; while pgrep -f measure_l4.sh > /dev/null; do sleep 30; done
B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; P=/root/venv/bin/python; IMG=/data/datasets/coco/images/val2017; D=/data/tmp/l4-row0; L=$D/l4_p2.log; : > $L
P2=/data/tmp/ds-yolo/phase2/onnx
for m in yolo26m_512:512 yolo26m_768:768 yolo26s_768:768 if_depth_ml:640 if_scale4:640; do n=${m%%:*}; sz=${m##*:}; [ -f $P2/$n.metadata.yaml ] && [ -s $P2/$n.metadata.yaml ] || sed "s/^- 640$/- $sz/" $D/onnx/yolo26m.metadata.yaml > $P2/$n.metadata.yaml
  if ! ls $P2/$n-*l4*-fp16.engine >/dev/null 2>&1; then t0=$(date +%s); timeout 3000 $BIN -m $P2/$n.onnx -s $IMG -b trt --precision fp16 --no-save --quiet --limit 5 > $D/out/build_$n.log 2>&1
    echo "== build $n: exit=$? $(grep -E '\[trt\] built' $D/out/build_$n.log | sed -E 's/ -> .*\//  /' | cut -c1-110) ($(( $(date +%s) - t0 )) s)" >> $L; fi
done
BASE=$(ls $D/onnx/yolo26m-*l4*-fp16.engine | head -1); L4E=$(ls $D/onnx/yolo26l-*l4*-fp16.engine | head -1)
for r in 1 2 3 4 5; do
  $P $D/trt_time.py $BASE 2>&1 | grep TRTTIME >> $L; $P $D/trt_time.py $L4E 2>&1 | grep TRTTIME | sed "s/TRTTIME [^ ]*/TRTTIME yolo26l/" >> $L
  for n in yolo26m_512 yolo26m_768 yolo26s_768 if_depth_ml if_scale4; do e=$(ls $P2/$n-*l4*-fp16.engine 2>/dev/null | head -1); [ -n "$e" ] && $P $D/trt_time.py $e 2>&1 | grep TRTTIME | sed "s/TRTTIME [^ ]*/TRTTIME $n/" >> $L; done
done
IN="/data/tmp/ds-yolo/phase1b/onnx/in_branch0.npy /data/tmp/ds-yolo/phase1b/onnx/in_branch1.npy /data/tmp/ds-yolo/phase1b/onnx/in_branch2.npy /data/tmp/ds-yolo/phase1b/onnx/in_branch3.npy"
cp /data/tmp/ds-yolo/phase1b/trt_time_in.py /data/tmp/ds-yolo/phase2/trt_prof_in.py $D/
for n in if_depth_ml if_scale4 yolo26m; do e=$(ls $P2/$n-*l4*-fp16.engine $D/onnx/$n-*l4*-fp16.engine 2>/dev/null | head -1); for r in 1 2; do $P $D/trt_time_in.py $e --inputs $IN --tag ${n}_real 2>&1 | grep TRTIN >> $L; done; $P $D/trt_prof_in.py $e $IN 2>&1 | grep -E "PROFIN|rror" >> $L; done
echo "DONE $(date -u +%T)" >> $L
