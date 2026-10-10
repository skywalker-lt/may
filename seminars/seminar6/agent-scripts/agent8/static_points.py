"""Static LBR operating points (base + 16-tile refiner always on) vs the measured T4 envelope. Ceiling = L640 detections
in the routed tiles (tiles_budget.log; L512 from seminar-5 agent 2). Refiner cost est.: seminar-5 lo/mid/hi + 0.10 split.
phi_req = share of the ceiling gain a trained expert must realise to reach envelope + 0.003 at that latency."""
import numpy as np
Ht=[1.65,2.75,3.78,4.88,5.32,6.89,12.41]; Ha=[0.4060,0.4795,0.5063,0.5262,0.5329,0.5417,0.5691]
env=lambda t: float(np.interp(t,Ht,Ha))
B={'L448':(4.28,0.5117,0.5381),'L512':(4.88,0.5262,0.5403),'L544':(5.32,0.5329,0.5426)}
for b,(t,a,c) in B.items():
    for nm,tr in [('lo',0.59),('mid',0.91),('hi',1.23)]:
        tt=t+tr; e=env(tt); req=(e+0.003-a)/(c-a)
        print(f'{b}+R16 {nm}: {tt:.2f} ms  ceiling {c:.4f}  env {e:.4f}  ceiling-env {c-e:+.4f}  phi_req {req:.2f}')
print('k-curve on L512 (ceiling): k=0 0.5262, 8 0.5375, 16 0.5403, 32 0.5414; random 16: 0.5284')
