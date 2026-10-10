#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/data/envs/rtdetr/bin/python
cd /data/tmp/ds-yolo/seminar6/work/agent8
date +%T; $P -I rescore_null.py > rescore_null.log 2>&1
date +%T; $P -I static_vs_late.py > static_vs_late.log 2>&1
date +%T; echo ALLDONE
