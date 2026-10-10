#!/bin/bash
export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python; cd /data/tmp/ds-yolo/seminar6/work/agent1
for args in "M640 L640 300 unc 0.1" "M640 L640 300 unc 1.0" "M640 L640 64 unc 0.1" "M640 L640 64 random 0.1" "M640 L640 64 random 0.1 1" "M640 L640 32 unc 0.1" "M640 L640 128 unc 0.1" "M640 L640 64 oracleE 0.1" "M640 L640 64 top 0.1" "M640 L640 64 band 0.1"; do $P qroute.py $args; done; echo DONE_A
