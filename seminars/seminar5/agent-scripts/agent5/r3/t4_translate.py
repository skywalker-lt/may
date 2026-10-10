"""Est.: translate the H200 stage split (prof_split.log, gather lowering) to the T4 and L4 with the packet's cost model.
H200 stage times are profiled ms scaled by the measured median delta (0.209 ms over tcr_m_base) / profiled sum (0.374).
T4 multipliers from kernels present in both the H200 stub profile and the T4 YOLO26-M profile: memory-bound decode
0.0398/0.0090 = 4.4x, TopK 0.0783/0.0294 = 2.7x, launch-bound gather 0.0046/0.0055 = 0.84x; conv blocks L16 0.435/0.139 = 3.1x,
L19 0.354/0.130 = 2.7x. GEMM rate on the T4 at 1,024 tokens: between 3.1 (L10 FFN, 400 tokens) and 11 GMAC/ms (L19 cv1,
1,600 tokens): 6-10 GMAC/ms. Fusion break (cost model): +0.014-0.04 ms per tapped layer, taps L2, L16, L19 (+L22 for TCR-3)."""
sc=0.209/0.3742
h={'router':0.0404*sc,'dispatch':0.0153*sc,'expert':0.2887*sc,'combine':0.0298*sc}
print('H200 median-scaled stage ms:',{k:round(v,3) for k,v in h.items()})
def t4(gmac, taps, small_kernels=15, mha=4):
    r=(0.03,0.08)                       # router: p(1-p) over 8400x80 (decode-like, 4.4x) + 3 reductions + TopK(100->16)
    d=(h['dispatch']*0.84+0.0, h['dispatch']*4.4+taps*0.04)   # gather kernel launch- to memory-bound, plus fusion breaks
    d=(max(d[0],0.01),d[1])
    e=(gmac/10+small_kernels*0.005+mha*0.01, gmac/6+small_kernels*0.010+mha*0.02)
    c=(0.02+0.02,0.07+0.05)             # mask/concat/decode of expert anchors + first TopK over 8400 -> 13.5k anchors
    tot=(r[0]+d[0]+e[0]+c[0], r[1]+d[1]+e[1]+c[1])
    return r,d,e,c,tot
for name,g,taps in [('stub (FFN 4x, 3.94 GMAC, 3 taps)',3.94,3),('TCR-3 (FFN 2x, 2.99 GMAC, 4 taps)',2.99,4),('TCR-3 lite (2 layers, 1.85 GMAC, 4 taps)',1.85,4)]:
    r,d,e,c,tot=t4(g,taps)
    print(f'T4 est. {name}: router {r[0]:.2f}-{r[1]:.2f} dispatch {d[0]:.2f}-{d[1]:.2f} expert {e[0]:.2f}-{e[1]:.2f} combine {c[0]:.2f}-{c[1]:.2f} total +{tot[0]:.2f}-{tot[1]:.2f} ms; mid +{(tot[0]+tot[1])/2:.2f}; x{(5.36+tot[0])/5.36:.2f}-{(5.36+tot[1])/5.36:.2f}')
    F=lambda t: 0.5261+0.0102*(t-5.36)
    print(f'   T4 packet bar {F(5.36+tot[0])+0.003:.4f}-{F(5.36+tot[1])+0.003:.4f}')
    # L4: compute part x0.47-0.56 of T4, launch-bound glue not reduced
    l4=(e[0]*0.47+ (r[0]+d[0]+c[0])*0.8, e[1]*0.56+(r[1]+d[1]+c[1])*1.0)
    FL=lambda t: 0.5261+0.0196*t
    print(f'   L4 est. +{l4[0]:.2f}-{l4[1]:.2f} ms; L4 packet bar {FL(l4[0])+0.003:.4f}-{FL(l4[1])+0.003:.4f}')
