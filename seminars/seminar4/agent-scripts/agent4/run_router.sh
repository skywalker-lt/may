#!/bin/bash
until grep -q CASCADE_DONE run_cascade.out; do sleep 10; done
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python router_cv.py > router_cv.log 2>&1
echo ROUTER_DONE
