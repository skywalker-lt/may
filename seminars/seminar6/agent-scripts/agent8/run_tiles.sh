#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/data/envs/rtdetr/bin/python; S=/data/tmp/ds-yolo/seminar6/work/agent8/tilemix8.py
cd /data/tmp/ds-yolo/seminar6/work/agent8
for a in "L544 L 0.16 Munc0.01 96" "L544 L 0.16 random 96" "L512 L 0.32 Munc0.01 96" "L512 L 0.08 Munc0.01 96" "L448 L 0.16 Munc0.01 96"; do
  date +%T; $P -I $S $a
done
echo DONE
