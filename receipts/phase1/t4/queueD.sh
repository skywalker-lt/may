#!/usr/bin/env bash
# Epoch-0 accuracy rows: val-protocol dumps (same command as the baseline table) rescored with pycocotools.
while pgrep -f "queueC.sh|queueB.sh|^bash ./ds_phase1.sh" > /dev/null; do sleep 20; done
B=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0; BIN=$B/yolomaster_edge; IMG=/root/coco/images/val2017; L=/root/ds/out/batchD.log
source /venv/main/bin/activate >/dev/null 2>&1; : > $L
for m in ${MODELS:-yolo26m_ref wb_top1_if wb_top1_matmul_local wb_top2_matmul_local}; do
  rm -rf /root/ds/dumpml_$m
  timeout 3600 $BIN -m /root/ds/onnx/$m.onnx -s $IMG -b trt --precision fp16 --conf 0.001 --iou 0.7 --multi-label --save-txt /root/ds/dumpml_$m --no-save --quiet 2>&1 | grep -E "^\[summary\]" | sed "s/^/  $m dumpml: /" | tee -a $L
  python /root/txt_to_coco_eval.py /root/ds/dumpml_$m 2>&1 | grep -E "images_with|PYCOCO" | sed "s/PYCOCO dumpml_/PYCOCOML /" | tee -a $L
done
echo "DONE $(date -u +%T)" | tee -a $L
