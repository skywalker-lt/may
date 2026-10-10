#!/bin/bash
cd /data/tmp/ds-yolo/seminar6/work/agent10
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/data/envs/rtdetr/bin/python probe_lwdetr_scale.py --size 512 --n 500 --stride 10 --out /data/tmp/ds-yolo/seminar6/work/agent10/lw500_512.npz > lw500_512.log 2>&1
echo "512 done $(date)"
/data/envs/rtdetr/bin/python probe_lwdetr_scale.py --size 640 --n 500 --stride 10 --out /data/tmp/ds-yolo/seminar6/work/agent10/lw500_640.npz > lw500_640.log 2>&1
echo "640 done $(date)"
