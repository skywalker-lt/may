#!/bin/bash
cd /data/tmp/ds-yolo/seminar6/work/agent4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data/envs/rtdetr/bin/python s1_bn_adapt.py > s1_bn_adapt.log 2>&1 && /data/envs/rtdetr/bin/python s2_eval_matrix.py 60 > s2_eval_matrix.log 2>&1
