#!/bin/bash
export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python; cd /data/tmp/ds-yolo/seminar6/work/agent1
for args in "M640 M640o2m 300 unc 0.1" "M640 M640o2m 16 unc 0.1" "M640 M640o2m 16 random 0.1" "M640 M640o2m 64 unc 0.1" "M640 M640o2m 64 random 0.1" "M640 M640o2m 300 unc 1.0"; do $P qroute.py $args; done; echo DONE_E
