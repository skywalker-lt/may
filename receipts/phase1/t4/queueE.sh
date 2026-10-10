#!/usr/bin/env bash
while pgrep -f "queueD.sh|queueC.sh|^bash ./ds_phase1.sh" > /dev/null; do sleep 20; done
./ds_phase1.sh "res_scale_out res_lowrank_out" 5 batchE > out/batchE.out 2>&1
