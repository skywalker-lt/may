#!/bin/bash
# sequential, one thread: wait for calibration, build the Q/DQ models, run the 500-image val subset three ways
export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python; cd /data/tmp/ds-yolo/seminar5/work/agent10/r2
$P make_qdq.py /data/tmp/l4-row0/onnx/yolo26m.onnx amax.json all q8_all.onnx
$P make_qdq.py /data/tmp/l4-row0/onnx/yolo26m.onnx amax.json stem q8_stem.onnx
$P run_val.py /data/tmp/l4-row0/onnx/yolo26m.onnx fp32 500 taps
$P run_val.py q8_all.onnx q8all 500
$P run_val.py q8_stem.onnx q8stem 500
echo PIPELINE DONE
