"""Discrete-event queue: one T4, Poisson (or bursty) arrivals, end-to-end deadline D per frame.
Policies: static rung; budget-routed ladder (slack shared over the backlog); frames whose deadline cannot be met are dropped.
AP of a trace = sum(share_r * AP_r) with dropped frames at 0 (pycocotools check: 20.5% dropped L544 -> 0.4247 = 0.797*0.5329).
T4 latencies: TensorRT-native unset-buffer medians. L480 AP interpolated (est.).
"""
import numpy as np, sys
ladder=[('M448',3.33,0.4937),('L448',4.28,0.5117),('L480',4.61,0.5190),('L512',4.88,0.5262),('L544',5.32,0.5329),('L576',6.25,0.5368),('L640',6.89,0.5417)]
T={k:t for k,t,a in ladder}; A={k:a for k,t,a in ladder}
Ht=[1.65,2.75,3.78,4.88,5.32,6.89,12.41]; Ha=[0.4060,0.4795,0.5063,0.5262,0.5329,0.5417,0.5691]
env=lambda t: float(np.interp(t,Ht,Ha))
def arrivals(n,mean_ia,burst,seed):
    r=np.random.default_rng(seed)
    if burst is None: return np.cumsum(r.exponential(mean_ia,n))
    ia=np.empty(n); state=0
    for i in range(n):
        if r.random()<0.01: state=1-state
        ia[i]=r.exponential(mean_ia*(burst if state else 1.0))
    return np.cumsum(ia)
def run(policy,arr,D,top='L640'):
    cand=[x for x in ladder if T[x[0]]<=T[top]]
    n=len(arr); free=0.0; i=0; counts={}; miss=0; lat=[]
    while i<n:
        now=max(free,arr[i])
        # drop frames already past their deadline
        while i<n and arr[i]+D<now+T['M448']: miss+=1; i+=1
        if i>=n: break
        now=max(free,arr[i])
        q=int(np.searchsorted(arr,now,side='right')-i)   # frames waiting incl. this one
        # slack-sharing budget: every waiting frame must still meet its deadline if served in order at this budget
        waits=[(arr[i+j]+D-now)/(j+1) for j in range(q)]
        budget=min(waits)
        rung=policy(cand,budget,q)
        if rung is None or now+T[rung]>arr[i]+D:
            miss+=1; i+=1; continue
        free=now+T[rung]; counts[rung]=counts.get(rung,0)+1; lat.append(free-arr[i]); i+=1
    sh={k:v/n for k,v in counts.items()}; mf=miss/n
    ap=sum(sh[k]*A[k] for k in sh); avg=sum(sh[k]*T[k] for k in sh)/max(1e-9,1-mf)
    return sh,mf,ap,avg,(np.mean(lat) if lat else float('nan'))
def static(r): return lambda cand,b,q: r
def routed(cand,b,q):
    ok=[x for x in cand if T[x[0]]<=b]
    return max(ok,key=lambda x:T[x[0]])[0] if ok else None
n=100000
print('scenario | best static (rung, miss%, AP) | ladder<=L544 (miss%, AP, avg ms, env(avg)) | ladder<=L640 (miss%, AP, avg ms, env(avg)) | ladder shares')
for mean_ia,D,burst in [(6.0,20.0,None),(5.6,20.0,None),(6.0,12.0,None),(7.0,20.0,None),(8.0,20.0,None),(7.0,15.0,3.0),(8.0,20.0,4.0),(6.5,33.0,None),(10.0,33.0,None)]:
    arr=arrivals(n,mean_ia,burst,1)
    best=None
    for r in ['M448','L448','L512','L544','L576','L640']:
        sh,mf,ap,avg,_=run(static(r),arr,D)
        if best is None or ap>best[2]: best=(r,mf,ap)
    out=[]
    for top in ['L544','L640']:
        sh,mf,ap,avg,ml=run(routed,arr,D,top); out.append((mf,ap,avg,env(avg),sh))
    tag=f'ia {mean_ia} ms{" x"+str(burst)+" bursts" if burst else ""}, D {D} ms, rho(L544) {5.32/mean_ia:.2f}'
    print(f'{tag} | {best[0]} {best[1]*100:.1f}% {best[2]:.4f} | {out[0][0]*100:.1f}% {out[0][1]:.4f} {out[0][2]:.2f} {out[0][3]:.4f} | {out[1][0]*100:.1f}% {out[1][1]:.4f} {out[1][2]:.2f} {out[1][3]:.4f} | '+' '.join(f'{k}:{v:.2f}' for k,v in sorted(out[1][4].items(),key=lambda kv:-kv[1])))
