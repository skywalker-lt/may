#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent5/r2; export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python
while read a; do $P mix_p3.py $a 2>&1 | grep -E 'P3MIX|Error|Traceback' >> p3mix.log; done <<'Q'
M L 0.16 Munc0.01 64
M L 0.16 random 64
M L 0.16 Munc0.01 32
M L 0.16 Munc0.01 96
M L 0.16 Munc0.01 9999
M M768 0.16 Munc0.01 64
M M768 0.16 random 64
M M768 0.16 Munc0.01 32
M X 0.16 Munc0.01 64
M X 0.16 random 64
Y11 Y12 0.16 Munc0.01 9999
Y11 Y12 0.16 random 9999
Y11 Y12 0.16 Munc0.01 64
Y11 Y12 0.40 Munc0.01 9999
Y11 Y12 0.40 random 9999
Q
echo DONE >> p3mix.log
