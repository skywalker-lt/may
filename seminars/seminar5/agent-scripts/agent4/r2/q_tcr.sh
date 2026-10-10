P=/data/envs/rtdetr/bin/python
export OMP_NUM_THREADS=1
$P tcr_scale.py M L 10 0.16 Munc0.01 64
$P tcr_scale.py M L 10 1.00 random 64
$P tcr_scale.py M X 10 0.16 Munc0.01 64
$P tcr_scale.py M L 10 0.16 random 64
$P tcr_scale.py M L 10 0.16 Munc0.01 96
$P tcr_scale.py M L 10 0.16 Munc0.01 32
$P tcr_scale.py M X 10 1.00 random 64
echo DONE
