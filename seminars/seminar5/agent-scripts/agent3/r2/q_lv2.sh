cd /data/tmp/ds-yolo/seminar5/work/agent3/r2
while pgrep -f "levelvar.py calib" > /dev/null; do sleep 5; done
env OMP_NUM_THREADS=1 /data/envs/rtdetr/bin/python levelvar.py run 0 5000 5 > lv_run.log 2>&1
