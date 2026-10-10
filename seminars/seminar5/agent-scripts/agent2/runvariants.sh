#!/bin/sh
cd /data/tmp/ds-yolo/seminar5/work/agent2
export OMP_NUM_THREADS=1
P=/data/envs/rtdetr/bin/python
$P evalcache.py /data/tmp/ds-yolo/seminar5/work/agent2/boxmix64_Lsmall.json /data/tmp/ds-yolo/seminar5/work/agent2/boxmix64_Lmedlarge.json >> evalcache.log 2>&1
BAND=64 FEAT=m640 COSTS=0.24,0.97 $P -u allocv.py 1e-6 2.5e-6 > alloc64_tailcost_m640.log 2>&1
BAND=64 FEAT=n320 COSTS=0.68,0.85 RCOST=0.179 $P -u allocv.py 1e-6 2.5e-6 > alloc64_fullcost_n320.log 2>&1
echo ALLDONE >> alloc64_fullcost_n320.log
