"""Pred.: dense envelope over (model, input scale) on the T4 at TCR's estimated latency. AP: measured public dumps
(M 448/512/576/608/640/768, L 448/512/576/640); L's curve quadratic in ln s through its 4 measured points.
T4 latency (est., no T4 pod): two models, (A) a + b (s/640)^2 with M's fixed term fitted on M's measured 512/640 unset
times 3.78/5.36 and L scaled; (B) pure pixel scaling t640 (s/640)^2. Bars: front + 0.003 and envelope + 0.003."""
import numpy as np
F=lambda t: 0.5261+0.0102*(t-5.36)
Ls=np.array([448,512,576,640]); Lap=np.array([0.5117,0.5262,0.5368,0.5417])
c=np.polyfit(np.log(Ls/640),Lap,2); Lfit=lambda s: np.polyval(c,np.log(s/640))
print('L fit residuals', np.round(Lap-Lfit(Ls),4), ' L@544 pred %.4f  L@608 pred %.4f'%(Lfit(544),Lfit(608)))
b=(5.36-3.78)/(1-0.64); a=5.36-b
def tL(s,mod): return (a+(6.89-a)*(s/640)**2) if mod=='A' else 6.89*(s/640)**2
ss=np.arange(448,641,32)  # input sizes are multiples of the max stride 32
for mod in 'AB':
    print(f'-- latency model {mod}: L@512 {tL(512,mod):.2f}, L@544 {tL(544,mod):.2f}, L@576 {tL(576,mod):.2f}, L@608 {tL(608,mod):.2f} ms (est.)')
    for d in [0.53,0.78,1.13]:
        t=5.36+d; ok=[s for s in ss if tL(s,mod)<=t]; s=max(ok); env=max(Lfit(s),0.5261)
        print(f'   TCR +{d:.2f} ms -> {t:.2f} ms: packet bar {F(t)+0.003:.4f} (+{F(t)+0.003-0.5261:.4f} over M); envelope = L@{s} {env:.4f}, bar {env+0.003:.4f} (+{env+0.003-0.5261:.4f} over M)')
