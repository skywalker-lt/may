#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent5; export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python
until grep -q DONE mix3.log; do sleep 20; done
while read a; do $P mix_eval.py $a 2>&1 | grep -E 'MIX|ALONE|Error' >> mix4.log; done <<'Q'
M X 10 0.16 Mlow0.01
M S 10 0.16 Mdet0.01
Q
echo DONE >> mix4.log
