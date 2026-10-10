#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent3/r3
export OMP_NUM_THREADS=1; PY=/data/envs/rtdetr/bin/python
$PY build_ei3.py /data/tmp/ds-yolo/seminar5/inputs/dumps_r2/dumpml_yolo26l_512_coco.json l512 512 noP3_16,noP3_32 > ei_l512b.log 2>&1
$PY mix3.py lvl:l512 > mix_lvl_l512.log 2>&1
$PY mix3.py rs2l > mix_rs2l.log 2>&1
