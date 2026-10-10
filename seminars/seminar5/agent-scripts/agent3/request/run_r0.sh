#!/usr/bin/env bash
# Agent 3 (seminar 5) R0 on the H200 pod. Part 1 needs no training (three CPU-built ONNX graphs, minutes);
# part 2 trains B and its same-recipe control A* on the frozen public trunk (2 x 10 epochs, two runs sharing one GPU);
# part 3 dumps the trained checkpoints under the protocol. Safe to re-run: runs resume, dumps are skipped if present.
#   bash run_r0.sh dumps | train | dumps-trained
source /root/anaconda3/etc/profile.d/conda.sh; conda activate yolo_master
cd /data/YOLO-Master; export PYTHONPATH=/data/YOLO-Master
R=/data/tmp/ds-yolo/seminar5/work/agent3/request          # this directory (copy it to the pod at the same path)
BIN=/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0/yolomaster_edge
IMG=/data/datasets/coco/images/val2017; ANN=/data/datasets/coco/annotations/instances_val2017.json
E=/data/runs_receipts/agent3; S=/training_data/runs_receipts/agent3; mkdir -p $E $S; L=$E/dump.log
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $L; }
dump_onnx() {  # <name> <onnx with .metadata.yaml sidecar>   (the dump step of scripts/ds_yolo/dump_batch.sh)
  n=$1; [ -f $S/dumpml_${n}_coco.json ] && return; cp $2 $E/$n.onnx; cp ${2%.onnx}.metadata.yaml $E/$n.metadata.yaml
  rm -rf $E/dumpml_$n
  timeout 3600 $BIN -m $E/$n.onnx -s $IMG -b trt --precision fp16 --conf 0.001 --iou 0.7 --multi-label --save-txt $E/dumpml_$n --no-save --quiet 2>&1 | grep -E "^\[summary\]|\[trt\] built" | sed "s/^/  $n: /" | tee -a $L
  python scripts/ds_yolo/txt_to_coco_eval.py $E/dumpml_$n --ann $ANN 2>&1 | grep -E "images_with|PYCOCO" | sed "s/^/  $n: /" | tee -a $L
  cp $E/dumpml_${n}_coco.json $S/ 2>/dev/null; }
case "$1" in
dumps)   # part 1: public trunk, no training
  dump_onnx m640_noP3head_o2o $R/m640_noP3head_o2o.onnx     # P3 anchors removed from the shipped NMS-free head
  dump_onnx m640_noP3head_o2m $R/m640_noP3head_o2m.onnx     # one-to-many head, P3 anchors removed, CLI NMS
  dump_onnx m640_bconst_o2o  $R/m640_bconst_o2o.onnx        # branch B at epoch 0 (= b_init.pt)
  cp $L $S/ ;;
train)   # part 2: about 2.5 H200 h for both runs together (est.)
  P=runs_agent3; mkdir -p /data/$P /training_data/$P
  launch() { n=$1; shift
    pgrep -f "train_branch.py .*--name $n\b" > /dev/null && { echo "$n already running"; return; }
    setsid nohup python $R/train_branch.py --project $P --name $n --epochs 10 --every 5 --mem-fraction 0.48 "$@" >> /data/$P/$n.out 2>&1 < /dev/null & disown
    echo "launched $n"; sleep 20; }
  launch b10 --init $R/b_init.pt --freeze 14                     # B: P4 out, P5 path and P4/P5 heads retrained, no P3
  launch astar10 --init /data/weights/yolo26m.pt --freeze 18     # A*: the same layers retrained WITH P3 (recipe control)
  ;;
dumps-trained)  # part 3: forced dumps of the trained branches (the protocol's own exporter, via dump_batch.sh)
  printf "b10_last_640 /data/runs_agent3/b10/weights/last.pt 640 o2o\nastar10_last_640 /data/runs_agent3/astar10/weights/last.pt 640 o2o\n" > $E/spec_trained.txt
  bash scripts/ds_yolo/dump_batch.sh $E/spec_trained.txt agent3 ;;
esac
