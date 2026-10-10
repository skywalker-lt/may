#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent5/r2; export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python
until grep -q DONE p3mix.log; do sleep 20; done
while read a; do $P mix_p3.py $a 2>&1 | grep -E 'P3MIX|Error|Traceback' >> p3mix_b.log; done <<'Q'
M L 0.10 Munc0.01 96
M L 0.16 random 96
M X 0.16 Munc0.01 96
M M768 0.16 Munc0.01 96
Q
echo DONE >> p3mix_b.log
