#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent6/r2; export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
for s in 0.8 1.5; do /data/envs/rtdetr/bin/python uniform_gap.py $s >> uniform_gap.log 2>&1; done
echo ALLDONE >> uniform_gap.log
