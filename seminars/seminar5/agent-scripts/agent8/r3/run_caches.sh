export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=/data/envs/rtdetr/bin/python; S=/data/tmp/ds-yolo/seminar4/inputs/receipts_h200; R=/data/tmp/ds-yolo/seminar5/inputs/dumps_r2
cd /data/tmp/ds-yolo/seminar5/work/agent8/r3
$P cache_abs.py m640o2o_h200 $S/dumpml_yolo26m_640_coco.json
$P cache_abs.py m640o2m $R/dumpml_yolo26m_640_o2m_coco.json
$P cache_abs.py spl_o2o $S/dumpml_splice10_last_640_coco.json
$P cache_abs.py spl_o2m $R/dumpml_splice10_last_640_o2m_coco.json
$P cache_abs.py ts768_o2o $S/dumpml_twoscale20_last_768_coco.json
$P cache_abs.py ts768_o2m $R/dumpml_twoscale20_last_768_o2m_coco.json
echo ALLDONE
