#!/bin/bash
# one process at a time, one thread each
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=/data/YOLO-Master
P=/data/envs/rtdetr/bin/python; R=/data/tmp/ds-yolo/seminar5/work/agent2/r2; Q=/data/tmp/ds-yolo/seminar5/work/agent2/request
cd $R
$P fwd.py /data/yolo-quant-work/weights/yolo26m.pt m 4900 5000 feat17 > log_m_const.log 2>&1
$P fwd.py /data/yolo-quant-work/weights/yolo26m.pt m 0 400 full,noP3,const17 --const m_const17.pt > log_m_400.log 2>&1
$P fwd.py $Q/yolo26l_land_b0.pt lb0 0 250 full > log_lb0_250.log 2>&1
$P fwd.py $Q/yolo26l_land_b3.pt lb3 0 250 full > log_lb3_250.log 2>&1
echo CHAIN DONE > chain.done
