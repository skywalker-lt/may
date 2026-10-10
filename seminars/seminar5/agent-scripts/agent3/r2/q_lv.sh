cd /data/tmp/ds-yolo/seminar5/work/agent3/r2
while ! grep -q QDONE build_l.log 2>/dev/null; do sleep 5; done
P="env OMP_NUM_THREADS=1 /data/envs/rtdetr/bin/python"
$P levelvar.py calib 200 > calib.log 2>&1
$P levelvar.py run 0 5000 3 > lv_run.log 2>&1
