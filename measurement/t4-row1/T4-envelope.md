# T4 receipt: the dense (model, input scale) envelope and the two-scale conditional engines (2026-10-09)

Tesla T4, driver 595.91.07, the unchanged yolo-master-edge 1.2.0 trt10-cuda12 bundle, TensorRT 10.16.1 fp16, batch 1, static
engines, TensorRT-native median of 500 iterations, five interleaved rounds, YOLO26-M at 640 re-timed in every round
(5.29-5.35 ms; the earlier table's 5.37). AP: multi-label pycocotools on full val2017 from the H200 dumps (device-independent).
Log and per-layer profiles: `/data/tmp/t4-row1/results/`. Unset-buffer basis unless marked "real".

| model | AP | T4 ms (median of five rounds) | over F(L) + 0.003 |
|---|---|---|---|
| YOLO26-M at 448 | 0.4937 | 3.33 | -0.0117 |
| YOLO26-M at 512 (earlier) | 0.5063 | 3.78 | -0.0067 |
| YOLO26-L at 448 | 0.5117 | 4.28 | -0.0064 |
| YOLO26-L at 480 | not dumped | 4.61 | - |
| YOLO26-L at 512 | 0.5262 | 4.88 | +0.0020 |
| YOLO26-M at 576 | 0.5203 | 4.93 | -0.0044 |
| YOLO26-M at 608 | 0.5234 | 5.20 | -0.0041 |
| YOLO26-L at 544 | 0.5329 | 5.32 | **+0.0042** |
| YOLO26-M at 640 | 0.5261 | 5.34 | -0.0028 |
| YOLO26-L at 576 | 0.5368 | 6.25 | -0.0014 |
| YOLO26-L at 640 (earlier) | 0.5417 | 6.89 | -0.0030 |

Readings.
1. **The envelope at YOLO26-M's latency is dense YOLO26-L at 544: 0.5329 at 5.32 ms**, +0.0068 AP over M at the same time.
   Interpolated to 5.36 ms it is about 0.5331, above the seminar's central estimate (0.5317) and near its pure-pixel bound.
   Any per-image route at M's latency must now reach 0.5361 to clear the envelope by 0.003.
2. **L's cost does not scale with pixels** below 640: L at 576 costs 6.25 ms (est. was 5.82), L at 544 5.32 (est. 5.32), L at
   512 4.88 (est. 4.86), L at 448 4.28 (est. 4.01). The envelope is flat above 5.3 ms (0.004 AP per ms from 544 to 576) and
   steep below it (0.024 AP per ms from 448 to 512).
3. **Whole YOLO26-N as a router costs 1.22 ms at 192 and 1.27 ms at 320** (launch-bound; the stem-only graph measured 0.18 ms
   at 320 in Phase 2). A router must be the stem alone, never the whole model.
4. **Branches inside the two-scale conditional pay a penalty on the T4.** On real inputs (dense M 5.03-5.07 ms real):
   rs2_fed (every branch through a Resize) 3.72 ms on the image routed to 512 and 5.30 on the image routed to 640;
   rs2_direct (the 640 branch read from the input) 3.78 and 5.52. Against the standalone engines (M at 512 3.47 real, M at
   640 5.05 real) each branch costs +0.25 to +0.27 ms Resize-fed and +0.31 to +0.49 ms directly fed. The earlier +0.65 ms
   datum is confirmed in kind and halved by the Resize feed, not removed. At the front's slope (0.0102 AP per ms) a
   +0.26 ms branch penalty is worth 0.0027 AP, which is the size of every surviving routed-shrink margin in seminar 5.
5. Consequence for the resolution line: RS-2 (M 512/640, share 0.5) averages 4.51 ms real against dense M's 5.05, a saving
   of 0.54 ms, not the 0.79 ms the seminar assumed; its AP 0.5230 sits 0.003 under the envelope (L at 448 to 512) at that
   latency. The deep shrink M 448/608 (0.5190 at an estimated 4.27 + 0.26 ms) sits about 0.001 under the envelope + 0.003
   once the branch penalty is charged. No per-image route on file clears the measured envelope by 0.003.
