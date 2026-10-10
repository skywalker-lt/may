while kill -0 24107
24110
25171 2>/dev/null; do sleep 10; done
OMP_NUM_THREADS=1 /data/envs/rtdetr/bin/python routes4.py noP3_32 noP5_128 P4only_32_128 > routes4.log 2>&1
OMP_NUM_THREADS=1 /data/envs/rtdetr/bin/python build_evalimgs.py dumpml_yolo26m_768_coco.json m768 full > build_m768.log 2>&1
echo QDONE >> routes4.log
