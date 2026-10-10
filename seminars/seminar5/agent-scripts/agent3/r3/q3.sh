#!/bin/bash
# sequential, one thread: wait for the L576 job, then L512 full
while pgrep -f "build_ei3.py .*l576" >/dev/null; do sleep 20; done
OMP_NUM_THREADS=1 /data/envs/rtdetr/bin/python build_ei3.py /data/tmp/ds-yolo/seminar5/inputs/dumps_r2/dumpml_yolo26l_512_coco.json l512 512 full > ei_l512.log 2>&1
