export OMP_NUM_THREADS=1
/data/envs/rtdetr/bin/python merge_exch.py > merge_exch.log 2>&1
/data/envs/rtdetr/bin/python cleaf.py > cleaf.log 2>&1
echo DONE > q3.done
