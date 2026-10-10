#!/bin/bash
# max-calibrated variants (the best per-tensor PTQ in the sensitivity scan) on the same 500-image subset
export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python; cd /data/tmp/ds-yolo/seminar5/work/agent10/r2
$P make_qdq.py /data/tmp/l4-row0/onnx/yolo26m.onnx amax.json all q8max_all.onnx max
$P make_qdq.py /data/tmp/l4-row0/onnx/yolo26m.onnx amax.json stem q8max_stem.onnx max
$P run_val.py q8max_all.onnx q8maxall 500
$P run_val.py q8max_stem.onnx q8maxstem 500
echo PIPELINE2 DONE
