#!/bin/bash
cd /data/tmp/ds-yolo/seminar4/work/agent4
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master
P=/data/envs/rtdetr/bin/python
$P oracle.py > oracle.log 2>&1 && $P cascade.py > cascade.log 2>&1 && $P router_cv.py > router_cv.log 2>&1
echo ALL_DONE >> oracle.log
