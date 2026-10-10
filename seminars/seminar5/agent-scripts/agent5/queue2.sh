#!/bin/bash
cd /data/tmp/ds-yolo/seminar5/work/agent5; export OMP_NUM_THREADS=1; P=/data/envs/rtdetr/bin/python
while read a; do $P mix_eval.py $a 2>&1 | grep -E 'MIX|ALONE|Error' >> mix2.log; done <<'Q'
M L 10 0.16 random
M L 10 0.16 Mdet0.01
M L 10 0.16 GTcount
M L 10 0.40 random
M L 10 0.40 Mdet0.01
M X 10 0.16 random
M X 10 0.16 Mdet0.01
M X 10 0.40 Mdet0.01
M X 10 0.40 random
Q
echo DONE >> mix2.log
