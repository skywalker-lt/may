#!/bin/bash
# Sequential, one process at a time, one thread each.
cd /data/tmp/ds-yolo/seminar6/work/agent5
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
F='^(loading|Done|creating|index|Loading|DONE)'
/data/envs/rtdetr/bin/python ladder.py 2>&1 | grep --line-buffered -vE "$F" > ladder.log
/data/envs/rtdetr/bin/python crop_full.py 2>&1 | grep --line-buffered -vE "$F" > crop_full.log
/data/envs/rtdetr/bin/python extra.py 2>&1 | grep --line-buffered -vE "$F" > extra.log
/data/envs/rtdetr/bin/python isocount.py 2>&1 | grep --line-buffered -vE "$F" > isocount.log
echo CHAIN_DONE >> isocount.log
