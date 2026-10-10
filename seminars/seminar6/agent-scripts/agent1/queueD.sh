#!/bin/bash
export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python; cd /data/tmp/ds-yolo/seminar6/work/agent1
for args in "L544 L640 16 unc 0.1" "L544 L640 16 random 0.1" "L544 X640 16 unc 0.1" "L544 X640 16 random 0.1" "M640 L640 16 unc 0.1" "M640 L640 16 random 0.1" "L544 L640 8 unc 0.1" "L544 X640 32 unc 0.1" "L544 X640 32 random 0.1" "L544 L640 16 random 0.1 1"; do $P qroute.py $args; done; echo DONE_D
