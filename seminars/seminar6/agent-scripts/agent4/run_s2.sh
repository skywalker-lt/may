#!/bin/bash
cd /data/tmp/ds-yolo/seminar6/work/agent4
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
while pgrep -f s1_bn_adapt.py > /dev/null; do sleep 10; done
grep -q '^saved' s1_bn_adapt.log && /data/envs/rtdetr/bin/python s2_eval_matrix.py 60 > s2_eval_matrix.log 2>&1
