cd /data/tmp/ds-yolo/seminar5/work/agent3/r2
P="env OMP_NUM_THREADS=1 /data/envs/rtdetr/bin/python"
for m in alone cheap stack ladder; do $P compare.py $m > cmp_$m.log 2>&1; done
echo CDONE >> cmp_ladder.log
