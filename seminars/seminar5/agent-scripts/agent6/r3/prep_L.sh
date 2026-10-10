#!/bin/bash
export OMP_NUM_THREADS=1
P=/data/tmp/ds-yolo/seminar5/work/agent6/prep_eval.py; D=/data/tmp/ds-yolo/seminar5/inputs
for n in dumps/dump_yolo26l_coco dumps_r2/dumpml_yolo26l_448_coco dumps_r2/dumpml_yolo26l_512_coco dumps_r2/dumpml_yolo26l_576_coco dumps_r2/dumpml_yolo26m_576_coco dumps/dumpml_yolo26m_512_coco; do
  b=$(basename $n); [ -f ev_$b.pkl ] || /data/envs/rtdetr/bin/python $P $D/$n.json ev_$b.pkl 2>&1 | grep "base AP" | sed "s/^/$b /"
done
echo PREP_DONE
