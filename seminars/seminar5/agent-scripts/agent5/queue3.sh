#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent5; export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python
until grep -q DONE mix2.log; do sleep 20; done
while read a; do $P mix_eval.py $a 2>&1 | grep -E 'MIX|ALONE|Error' >> mix3.log; done <<'Q'
M L 10 0.16 Munc0.01
M X 10 0.16 Munc0.01
M L 10 0.10 Munc0.01
M L 10 0.10 random
M M768 10 0.16 Munc0.01
M M768 10 0.16 random
M L 10 0.16 random 1
M X 10 0.10 Munc0.01
M X 10 0.10 random
Q
echo DONE >> mix3.log
