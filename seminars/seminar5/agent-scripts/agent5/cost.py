"""Cost model (est.) for the tile-choice refiner on YOLO26-M. T4 conv throughput calibrated on the T4 profile of
YOLO26-M (layer 16: 3.46 GMAC in 0.416 ms; layer 19: 2.41 GMAC in 0.351 ms)."""
rate_lo,rate_hi=2.41/0.351,3.46/0.416           # GMAC per ms, T4
C=256; tok=64                                    # 8x8 P3 tokens per 64-px tile
fuse=(256+1024+512)*C                            # P3 neck + space-to-depth(P2) + upsampled P4 -> 256
layer=3*C*C+C*C+2*tok*C+2*C*2*C                  # window MHSA (64 tokens) + FFN x2
head=C*C+C*84*5                                  # 1x1 MLP head: one stride-8 anchor + four stride-4 sub-anchors per token
for nl in (2,4):
    per=fuse+nl*layer+head
    for k in (10,16):
        g=per*k*tok/1e9
        for eff in (0.6,0.8):
            pass
        t_lo=g/(rate_hi*0.8); t_hi=g/(rate_lo*0.6)
        ov_lo,ov_hi=0.08,0.15
        d_lo,d_hi=t_lo+ov_lo,t_hi+ov_hi
        bar=lambda d:0.5261+0.0102*d+0.003
        # L4: 0.47-0.56x of T4 for compute, plus ~20 extra small kernels at 4-6 us each (launch-bound)
        l_lo,l_hi=0.47*t_lo+0.08,0.56*t_hi+0.12+0.03
        barl=lambda d:0.5261+0.0196*d+0.003
        print(f'layers={nl} k={k}: expert {per/1e6:.2f} MMAC/token, {g:.2f} GMAC; T4 +{d_lo:.2f}..+{d_hi:.2f} ms -> bar {bar(d_lo):.4f}..{bar(d_hi):.4f} (need +{bar(d_lo)-0.5261:.4f}..+{bar(d_hi)-0.5261:.4f} over M)'
              f' | L4 +{l_lo:.2f}..+{l_hi:.2f} ms -> bar {barl(l_lo):.4f}..{barl(l_hi):.4f}')
# killed variant: P3-head skipping on the T4 (MAC-saving)
p3head=0.1306+0.2423   # profile, one2one cv2/cv3 P3 branch
for keep in (0.5,0.25):
    print(f'P3-head skip keep={keep}: saves <= {p3head*(1-keep):.3f} ms before dispatch overhead 0.08-0.15 ms; worth <= {0.0102*p3head*(1-keep):.4f} AP on the T4 front')
