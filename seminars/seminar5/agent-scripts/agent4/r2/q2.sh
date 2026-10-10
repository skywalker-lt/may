export OMP_NUM_THREADS=1
PYTHONPATH=/data/YOLO-Master /data/envs/rtdetr/bin/python build_p4attn.py > build_p4attn.log 2>&1
/data/envs/rtdetr/bin/python router_feats.py val > rfeat_val.log 2>&1
/data/envs/rtdetr/bin/python router_feats.py train > rfeat_train.log 2>&1
echo DONE > q2.done
