#!/bin/bash
until grep -q "n/m only at avg 2.75" oracle.log || grep -q Traceback oracle.log; do sleep 5; done
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python cascade.py > cascade.log 2>&1
echo CASCADE_DONE
