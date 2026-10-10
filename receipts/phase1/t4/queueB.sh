#!/usr/bin/env bash
while pgrep -f "^bash ./ds_phase1.sh" > /dev/null; do sleep 20; done
./ds_phase1.sh "wb_top1_if wb_top1_conv_local wb_top1_matmul_local wb_top2_conv_local wb_top2_matmul_local" 5 batchB > out/batchB.out 2>&1
