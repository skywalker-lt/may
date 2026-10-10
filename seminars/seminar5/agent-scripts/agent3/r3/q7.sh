#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent3/r3
export OMP_NUM_THREADS=1; PY=/data/envs/rtdetr/bin/python; R=/data/tmp/ds-yolo/seminar5/inputs/dumps_r2
while pgrep -f "q5.sh|q6.sh" >/dev/null; do sleep 15; done
$PY build_ei3.py $R/dumpml_m640_noP3head_o2o_coco.json ho 640 full > ei_ho.log 2>&1
$PY mix3.py dump:ho:m > mix_ho.log 2>&1
