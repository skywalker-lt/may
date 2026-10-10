#!/bin/bash
cd /data/tmp/ds-yolo/seminar6/work/agent4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data/envs/rtdetr/bin/python s14_occl_check.py > s14_occl_check.log 2>&1
/data/envs/rtdetr/bin/python s13_extend.py > s13_extend.log 2>&1
