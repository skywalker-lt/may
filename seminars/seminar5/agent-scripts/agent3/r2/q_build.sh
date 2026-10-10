cd /data/tmp/ds-yolo/seminar5/work/agent3/r2
P="env OMP_NUM_THREADS=1 /data/envs/rtdetr/bin/python"
$P build2.py dumpml_yolo26m_512_coco.json m512 512 full,noP3_16,noP3_32 > build_m512.log 2>&1
$P build2.py dump_yolo26l_coco.json l 640 full > build_l.log 2>&1
echo QDONE >> build_l.log
