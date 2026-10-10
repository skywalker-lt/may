#!/bin/bash
# wait until the RT-DETR probe has saved 150 images, stop it, then run the LW-DETR probe (one thread throughout)
cd /data/tmp/ds-yolo/seminar6/work/agent10
until [ -f probe250.npz ] && /data/envs/rtdetr/bin/python -c "import numpy as np,sys; sys.exit(0 if int(np.load('probe250.npz')['n'])>=150 else 1)" 2>/dev/null; do sleep 30; done
pkill -f probe_rtdetr_depth_r1.py; sleep 2; cp probe250.npz rt150.npz
echo "rt150 saved $(date)"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /data/envs/rtdetr/bin/python probe_lwdetr_depth.py --n 200 --stride 25 --out /data/tmp/ds-yolo/seminar6/work/agent10/lw200.npz > lw200.log 2>&1
echo "lw done $(date)"
