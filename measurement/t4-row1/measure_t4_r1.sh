#!/usr/bin/env bash
# T4 session for the seminar-5 envelope: public L at 448/480/512/544/576, M at 448/576/608, N at 192/320 (router cost bound),
# the Resize-fed and directly-fed two-scale If engines on real inputs; dense M re-timed in every round. Same protocol as T4-baselines.md:
# unchanged 1.2.0 trt10-cuda12 bundle, TensorRT 10.16.1 fp16, batch 1, native median of 500 iterations, five rounds.
set -uo pipefail
B=${B:-/root/bundle/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0}; BIN=$B/yolomaster_edge; P=${P:-/venv/main/bin/python}
IMG=${IMG:-/root/coco/images/val2017}; D=/root/t4r1; X=$D/onnx; L=$D/t4_r1.log; mkdir -p $D/out $D/profiles
log() { echo "$*" | tee -a $L; }
log "host: $(nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,power.limit --format=csv,noheader) | $($BIN --version 2>&1 | head -1) | $(date -u +%FT%TZ)"
ORDER="yolo26m_640 yolo26l_448 yolo26l_480 yolo26l_512 yolo26l_544 yolo26l_576 yolo26m_448 yolo26m_576 yolo26m_608 yolo26n_192 yolo26n_320 rs2_fed rs2_direct"
for m in $ORDER; do f=$X/$m.onnx; [ -f $f ] || { log "== $m: missing"; continue; }
  if ! ls $X/$m-*-fp16.engine >/dev/null 2>&1; then t0=$(date +%s); timeout 3000 $BIN -m $f -s $IMG -b trt --precision fp16 --no-save --quiet --limit 5 > $D/out/build_$m.log 2>&1
    log "== build $m: exit=$? $(grep -E '\[trt\] built' $D/out/build_$m.log | sed -E 's/ -> .*\//  /' | cut -c1-110) ($(( $(date +%s) - t0 )) s)"; fi
done
eng() { ls $X/$1-*-fp16.engine 2>/dev/null | head -1; }
BASE=$(eng yolo26m_640)
for r in 1 2 3 4 5; do log "-- round $r $(date -u +%T) temp $(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader)"
  for m in $ORDER; do e=$(eng $m); [ -n "$e" ] || continue; log "$($P $D/trt_time.py $e 2>&1 | grep -E 'TRTTIME|rror' | sed "s/TRTTIME [^ ]*/TRTTIME $m/")"; done
done
for r in 1 2 3; do log "-- real-input round $r"; for m in yolo26m_640 rs2_fed rs2_direct; do e=$(eng $m); [ -n "$e" ] && log "$($P $D/trt_time_in.py $e --inputs $X/in640_shrink.npy $X/in640_keep.npy --tag $m 2>&1 | grep -E 'TRTIN|rror')"; done; done
for m in yolo26l_512 yolo26m_448 rs2_fed; do e=$(eng $m); [ -n "$e" ] && log "$($P $D/trt_time.py $e --iters 100 --profile $D/profiles/profile_$m.txt 2>&1 | grep PROFILE | sed "s/^/$m /")"; done
log "DONE $(date -u +%T)"
