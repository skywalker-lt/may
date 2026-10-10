#!/bin/bash
export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python; cd /data/tmp/ds-yolo/seminar6/work/agent1
$P qroute.py L544 L640 64 slot 0.1
$P qroute.py L544 X640 64 slot 0.1
$P qroute.py L544 X640 32 unc 0.1
RESCORE=1 $P qroute.py L544 X640 64 unc 0.1
RESCORE=1 $P qroute.py L544 L640 64 unc 0.1
$P qroute.py L544 L640 64 oracleE 0.1
echo DONE_F
