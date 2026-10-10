"""Agent 8, seminar 6, round 1. Budget-routing analysis on the measured T4 envelope.
All latencies: T4, TensorRT-native unset-buffer medians (T4-envelope.md, T4-baselines.md). AP: multi-label pycocotools
(N/S/X at 640: single-label, flagged). Nothing here is an L4 number.
"""
import numpy as np, json, sys
rng = np.random.default_rng(0)
# (name, T4 ms, AP)
pts = {
 'N640':(1.65,0.4060,'single-label'),'S640':(2.75,0.4795,'single-label'),
 'M448':(3.33,0.4937,''),'M512':(3.78,0.5063,''),'L448':(4.28,0.5117,''),'L480':(4.61,None,''),
 'L512':(4.88,0.5262,''),'M576':(4.93,0.5203,''),'M608':(5.20,0.5234,''),'L544':(5.32,0.5329,''),
 'M640':(5.34,0.5261,''),'L576':(6.25,0.5368,''),'L640':(6.89,0.5417,''),'X640':(12.41,0.5691,'single-label'),
}
P = sorted([(v[0],v[1],k) for k,v in pts.items() if v[1] is not None])
# upper concave hull (envelope): a random route realises any chord, so the dense null at an average latency is the hull
def hull(P):
    H=[]
    for t,a,k in P:
        while len(H)>=2:
            (t1,a1,_),(t2,a2,_)=H[-2],H[-1]
            if (a2-a1)*(t-t1) <= (a-a1)*(t2-t1): H.pop()   # H[-1] under chord H[-2]->new
            else: break
        H.append((t,a,k))
    # drop points dominated (lower AP at higher latency) from the right
    out=[]; best=-1
    for t,a,k in H:
        if a>best: out.append((t,a,k)); best=a
    return out
H = hull(P)
print('envelope vertices (T4 ms, AP):', [(t,a,k) for t,a,k in H])
Ht=np.array([h[0] for h in H]); Ha=np.array([h[1] for h in H])
def env(t):
    t=np.asarray(t,float); return np.where(t<Ht[0], np.nan, np.interp(t,Ht,Ha))
def env_step(t):
    """best dense point with latency <= t (what a per-frame budget actually buys: no chord, a point)."""
    t=np.asarray(t,float); out=np.full(t.shape,np.nan)
    for tt,aa,k in P:
        out=np.where(t>=tt, np.maximum(np.nan_to_num(out,nan=-1),aa), out)
    return out
print('\n== 1. Jensen: a per-frame budget b with mean 5.32 ms; E[env(b)] vs env(E b) (chord form and point form)')
for spread in [0.0,0.3,0.5,1.0,1.5,2.0]:
    b = 5.32 + np.where(rng.random(200000)<0.5,-spread,spread)
    print(f'  two-point +-{spread:.1f} ms: env(mean)={env(5.32):.4f}  E[env(b)] chord={np.nanmean(env(b)):.4f}  point={np.nanmean(env_step(b)):.4f}  Jensen loss (point) = {env(5.32)-np.nanmean(env_step(b)):+.4f}')
for lo,hi in [(4.3,6.3),(3.3,7.3),(4.9,5.8)]:
    b=rng.uniform(lo,hi,200000)
    print(f'  uniform[{lo},{hi}] (mean {np.mean(b):.2f}): env(mean)={float(env(np.mean(b))):.4f}  E[env(b)] chord={np.nanmean(env(b)):.4f}  point={np.nanmean(env_step(b)):.4f}')

print('\n== 2. Fixed-cost fit  ms = a + b*px^2 (est.), from the measured T4 points')
for fam,keys in [('M',['M448','M512','M576','M608','M640']),('L',['L448','L480','L512','L544','L576','L640'])]:
    px2=np.array([int(k[1:])**2 for k in keys],float); ms=np.array([pts[k][0] for k in keys])
    A=np.c_[np.ones_like(px2),px2]; coef,res,_,_=np.linalg.lstsq(A,ms,rcond=None)
    pred=A@coef
    print(f'  {fam}: a={coef[0]:.2f} ms fixed, b={coef[1]*1e6:.2f} ms per Mpx ; residuals '+' '.join(f'{k}:{m-p:+.2f}' for k,m,p in zip(keys,ms,pred)))
    # fit without 576 for L (the anomalous tactic) and without 640
    if fam=='L':
        sel=[0,1,2,3]; coef2,_,_,_=np.linalg.lstsq(A[sel],ms[sel],rcond=None)
        print(f'  L (448-544 only): a={coef2[0]:.2f}, b={coef2[1]*1e6:.2f}; pred L576={coef2[0]+coef2[1]*576**2:.2f} (meas 6.25), L640={coef2[0]+coef2[1]*640**2:.2f} (meas 6.89)')
        print(f'  est. batch-2 L544 if the fixed term is launch/latency-bound: {coef2[0]+2*coef2[1]*544**2:.2f} ms per pair = {(coef2[0]+2*coef2[1]*544**2)/2:.2f} ms/frame (est.; the profile shows 132 of 304 layers under 10 us, 0.67 ms, and 82 layers at 10-20 us, 1.20 ms)')

print('\n== 3. Four-width engine (M stem + s/m/l/x tails), measured T4 3.88/4.95/6.17/8.86 (M 5.33 same session): what each tail must score')
for name,t in [('s-tail',3.88),('m-tail',4.95),('l-tail',6.17),('x-tail',8.86)]:
    print(f'  {name} at {t:.2f} ms: envelope {float(env(t)):.4f}, bar (env+0.003) {float(env(t))+0.003:.4f}')

print('\n== 4. Queue simulation: Poisson arrivals, one T4, end-to-end deadline D; static rung vs budget-routed ladder')
ladder = [('M448',3.33,0.4937),('L448',4.28,0.5117),('L480',4.61,0.5190),('L512',4.88,0.5262),('L544',5.32,0.5329),('L576',6.25,0.5368),('L640',6.89,0.5417)]
# L480 AP not dumped: interpolated on the 448-512 chord, labelled est.
def simulate(policy, mean_ia, D, n=200000, seed=1, burst=None):
    r=np.random.default_rng(seed)
    if burst is None:
        ia = r.exponential(mean_ia, n)
    else:
        # two-state arrival process: calm (mean_ia) / burst (mean_ia*burst) with sticky states
        state=0; ia=np.empty(n)
        for i in range(n):
            if r.random()<0.02: state=1-state
            ia[i]= r.exponential(mean_ia if state==0 else mean_ia*burst)
    arr=np.cumsum(ia); free=0.0; counts={}; miss=0
    for i in range(n):
        start=max(arr[i],free); slack = arr[i]+D - start      # time left to the deadline when the GPU becomes free
        q = int(np.sum((arr[i:i+64] <= free)) )               # frames already waiting (approx, local window)
        rung = policy(slack, q)
        if rung is None or slack < rung[1]:
            miss+=1; continue                                  # dropped: no output for this frame (scores as empty)
        free = start + rung[1]; counts[rung[0]]=counts.get(rung[0],0)+1
    shares={k:v/n for k,v in counts.items()}; missf=miss/n
    ap_chord = sum(shares[k]*next(a for nm,t,a in ladder if nm==k) for k in shares)   # misses contribute 0 (checked with pycocotools, see mix log)
    avg_ms = sum(shares[k]*next(t for nm,t,a in ladder if nm==k) for k in shares)/(1-missf) if missf<1 else float('nan')
    return shares, missf, ap_chord, avg_ms
def static(rname):
    r=next(x for x in ladder if x[0]==rname)
    return lambda slack,q: r
def adaptive(maxr='L576'):
    cand=[x for x in ladder if x[1] <= next(y for y in ladder if y[0]==maxr)[1]]
    def pol(slack,q):
        ok=[x for x in cand if x[1]<=slack]
        return max(ok,key=lambda x:x[1]) if ok else None
    return pol
for mean_ia,D,burst in [(6.0,20.0,None),(5.6,20.0,None),(6.0,12.0,None),(7.0,15.0,3.0),(6.5,33.0,None)]:
    print(f'  -- Poisson mean inter-arrival {mean_ia} ms{" with 3x bursts" if burst else ""}, deadline {D} ms (load at L544 = {5.32/mean_ia:.2f})')
    for rn in ['M448','L448','L512','L544','L576']:
        sh,mf,ap,avg=simulate(static(rn),mean_ia,D,n=60000,burst=burst)
        print(f'     static {rn}: miss {mf*100:5.1f}%  AP (chord, misses=0) {ap:.4f}')
    for mx in ['L544','L576','L640']:
        sh,mf,ap,avg=simulate(adaptive(mx),mean_ia,D,n=60000,burst=burst)
        print(f'     budget-routed ladder up to {mx}: miss {mf*100:5.1f}%  AP {ap:.4f}  avg {avg:.2f} ms  shares '+' '.join(f'{k}:{v:.2f}' for k,v in sorted(sh.items(),key=lambda kv:-kv[1])))

print('\n== 5. Anytime bound: deadline d unknown at frame start. Static engine r scores AP_r if d>=t_r else 0.')
print('   Anytime (exit at tau with AP x, full at T with AP A): x*P(tau<=d<T) + A*P(d>=T). Break-even x against the best static choice:')
for lo,hi in [(4.5,6.5),(3.5,7.5),(5.0,6.0)]:
    d=rng.uniform(lo,hi,200000)
    best=max(((np.mean(d>=t)*a, k) for t,a,k in P), key=lambda z:z[0])
    for tau,T,A in [(3.33,5.32+0.3,0.5329),(4.28,5.32+0.3,0.5329),(3.5,5.32+0.3,0.5329)]:
        pmid=np.mean((d>=tau)&(d<T)); pfull=np.mean(d>=T)
        x_be=(best[0]-A*pfull)/pmid if pmid>0 else float('nan')
        print(f'   d~U[{lo},{hi}]: best static {best[1]} E[AP]={best[0]:.4f}; anytime exit@{tau} full@{T:.2f}: needs exit AP >= {x_be:.4f} (envelope at {tau} ms = {float(env_step(tau)):.4f})')
    print(f'   restart anytime M448 then L544 (output at 3.33, replaced at 8.65): E[AP] = {np.mean(d>=3.33)*0.4937+np.mean(d>=8.65)*(0.5329-0.4937):.4f}')
