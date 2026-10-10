#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent3/r3
export OMP_NUM_THREADS=1; PY=/data/envs/rtdetr/bin/python; R=/data/tmp/ds-yolo/seminar5/inputs/dumps_r2
while pgrep -f q4.sh >/dev/null; do sleep 15; done
$PY build_ei3.py $R/dumpml_m640_noP3head_o2m_coco.json hm 640 full > ei_hm.log 2>&1
$PY build_ei3.py $R/dumpml_yolo26m_640_o2m_coco.json am 640 full,noP3_16,noP3_32 > ei_am.log 2>&1
$PY build_ei3.py $R/dumpml_m640_bconst_o2o_coco.json bc 640 full > ei_bc.log 2>&1
$PY mix3.py dump:hm:am > mix_hm.log 2>&1
$PY mix3.py lvl:am > mix_lvl_am.log 2>&1
$PY mix3.py dump:bc:m > mix_bc.log 2>&1
