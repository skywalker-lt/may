#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/data/envs/rtdetr/bin/python; S=/data/tmp/ds-yolo/seminar5/work/agent2/r3/tilemix.py
cd /data/tmp/ds-yolo/seminar5/work/agent2/r3
for a in "M - 0 x 0" "L512 - 0 x 0" "L512 L 0.16 Munc0.01 96" "L512 L 0.16 random 96" "L512 L 0.16 Munc0.01 9999" "L512 L 0.16 Munc0.01 64" "L576 L 0.16 Munc0.01 96" "L576 L 0.16 random 96" "L512 X 0.16 Munc0.01 96" "M L 0.16 Munc0.01 96"; do
  $P $S $a
done
echo DONE
