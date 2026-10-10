"""Agent 8, round 2. Systems row for the two-engine form: deadline-miss rate and AP of (i) the single-engine always-on
refiner L512+R (5.79 ms est., ceiling AP 0.5403 or 0.5440 at X re-score), (ii) dense L512 alone, (iii) the late-bound
two-engine form, on the round-1 traces. Same simulator as queue_lbr.py (functions copied; ONE thread)."""
import numpy as np, itertools
n=20000
DENSE=[('M448',3.33,0.4937),('M512',3.78,0.5063),('L448',4.28,0.5117),('L480',4.61,0.5190),('L512',4.88,0.5262),
       ('L544',5.32,0.5329),('L576',6.25,0.5368),('L640',6.89,0.5417)]
T={k:t for k,t,a in DENSE}; A={k:a for k,t,a in DENSE}
def arrivals(mean_ia,burst,seed):
    r=np.random.default_rng(seed)
    if burst is None: return np.cumsum(r.exponential(mean_ia,n))
    st=(r.random(n)<0.01).cumsum()%2; return np.cumsum(r.exponential(mean_ia*np.where(st==1,burst,1.0)))
def budget(arr,i,now,D,extra=0.0,start=0):
    q=int(np.searchsorted(arr,now,side='right'))-i
    js=np.arange(start,max(q,start+1)); js=js[js<len(arr)-i]
    return np.min((arr[i+js]+D-now-extra)/(js+1-start)) if len(js) else np.inf, q
def run(arr,D,menu,alpha,top,refine=None,theta=None,seed=7):
    menu=[m for m in menu if m[1]<=top]; tmin=menu[0][1]
    free=0.0; tot=0.0; miss=0; nref=0; nbase=0; i=0; N=len(arr); lat=[]
    while i<N:
        now=max(free,arr[i])
        if now+tmin>arr[i]+D: miss+=1; i+=1; continue
        b,_=budget(arr,i,now,D)
        ok=[m for m in menu if m[1]<=alpha*b and now+m[1]<=arr[i]+D]
        if not ok: ok=[m for m in menu if now+m[1]<=arr[i]+D][:1]
        m=ok[-1]; now+=m[1]; ap=m[2]
        if refine and m[0] in refine:
            nbase+=1; tr,apr=refine[m[0]]
            bb,_=budget(arr,i,now,D,extra=tr,start=1)
            if now+tr<=arr[i]+D and bb>=theta:
                now+=tr; ap=apr; nref+=1
        tot+=ap; free=now; lat.append(now-arr[i]); i+=1
    lat=np.array(lat)
    return tot/N, miss/N, nref/max(nbase,1), np.percentile(lat,99)
scen=[(5.6,20.0,None),(6.0,12.0,None),(6.0,20.0,None),(7.0,20.0,None),(7.0,15.0,3.0),(8.0,20.0,4.0),(6.5,33.0,None)]
for apR,tag in [(0.5403,'L-taught ceiling'),(0.5440,'X re-score ceiling')]:
    tr=0.91; static=[('L512+R',4.88+tr,apR)]; REF={'L512':(tr,apR)}
    print(f'\n### refined point {tag}: {4.88+tr:.2f} ms, AP {apR}')
    print('scenario | static L512: miss% AP p99ms | static L512+R always on: miss% AP p99ms | late two-engine: miss% AP refine-share p99ms')
    for mean_ia,D,burst in scen:
        arr=arrivals(mean_ia,burst,1)
        s0=run(arr,D,[('L512',4.88,0.5262)],1.0,4.88); s1=run(arr,D,static,1.0,9.0)
        best=None
        for alpha,th in itertools.product([0.75,0.9,1.0],[4.3,4.9,5.4]):
            l=run(arr,D,[('L512',4.88,0.5262)],alpha,4.88,REF,th)
            if best is None or l[0]>best[0]: best=l
        print(f'ia {mean_ia}{"x"+str(burst) if burst else ""} D {D:.0f} | {s0[1]*100:.1f}% {s0[0]:.4f} {s0[3]:.1f} | {s1[1]*100:.1f}% {s1[0]:.4f} {s1[3]:.1f} | {best[1]*100:.1f}% {best[0]:.4f} {best[2]:.2f} {best[3]:.1f}',flush=True)
print('DONE')
