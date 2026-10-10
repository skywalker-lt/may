#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent6/r2; export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1; P=/data/envs/rtdetr/bin/python
while read a; do $P tcr_p3only.py $a 2>&1 | grep -E 'P3ONLY|Error|error' >> tcr_p3only.log; done <<'Q'
M L 10 0.16 Munc0.01 64
M L 10 0.16 random 64
M L 10 0.16 Munc0.01 32
M L 10 0.16 Munc0.01 96
M X 10 0.16 Munc0.01 64
M X 10 0.16 random 64
M M768 10 0.16 Munc0.01 64
M M768 10 0.16 random 64
M M768 10 0.16 Munc0.01 32
Q
echo DONE >> tcr_p3only.log
