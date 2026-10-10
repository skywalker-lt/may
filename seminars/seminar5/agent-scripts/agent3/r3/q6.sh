#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent3/r3
while pgrep -f q5.sh >/dev/null; do sleep 15; done
OMP_NUM_THREADS=1 /data/envs/rtdetr/bin/python combo.py > combo.log 2>&1
