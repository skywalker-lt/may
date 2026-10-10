"""Seminar 6, agent 8, round 1. Late-bound budget refinement (LBR) against the best scheduler over the dense envelope.
One T4 (TensorRT-native unset-buffer medians from T4-envelope.md); arrivals Poisson or two-state bursty; per-frame
end-to-end deadline D. AP of a trace = mean over frames of the AP of the operating point that served it, dropped
frames 0 (pycocotools check in mix_eval_misscheck.log: 20.5% dropped L544 -> 0.4247 = 0.795*0.5329).
Families (each with its policy parameters tuned on the same trace, best kept):
  dense : slack-shared budget b at frame start, largest dense point with cost <= alpha*b, capped at 'top'.
  early : the same rule over dense points plus (L base + refiner) points, committed at frame start.
  late  : base chosen as in 'dense' (cap top); after the base finishes, the refiner runs iff it meets this frame's
          deadline and leaves every waiting frame a slack-shared budget >= theta; the base output is the fallback.
  null  : 'late' with the refine decision replaced by a coin at the late family's realised refine share (the
          refiner is launched blind; if it would overrun, the base output is used and the GPU time is still spent).
Refiner points: AP_base + phi*(ceiling gain), ceiling from the tile mixtures (L640 detections in the 16 routed tiles).
Refiner T4 cost: seminar-5 estimate +0.49/+0.81/+1.13 ms (lo/mid/hi) plus 0.10 ms est. for the two-engine split.
L480's AP is interpolated (est.). Everything here is T4; no L4 number appears.
usage: queue_lbr.py [n_frames]
"""
import numpy as np, sys, itertools, json
n=int(sys.argv[1]) if len(sys.argv)>1 else 40000
DENSE=[('M448',3.33,0.4937),('M512',3.78,0.5063),('L448',4.28,0.5117),('L480',4.61,0.5190),('L512',4.88,0.5262),
       ('L544',5.32,0.5329),('L576',6.25,0.5368),('L640',6.89,0.5417)]
CEIL=json.load(open('/data/tmp/ds-yolo/seminar6/work/agent8/ceil.json'))   # base -> ceiling AP with 16 tiles
T={k:t for k,t,a in DENSE}; A={k:a for k,t,a in DENSE}
def arrivals(mean_ia,burst,seed):
    r=np.random.default_rng(seed)
    if burst is None: return np.cumsum(r.exponential(mean_ia,n))
    st=(r.random(n)<0.01).cumsum()%2; return np.cumsum(r.exponential(mean_ia*np.where(st==1,burst,1.0)))
def budget(arr,i,now,D,extra=0.0,start=0):
    q=int(np.searchsorted(arr,now,side='right'))-i
    js=np.arange(start,max(q,start+1)); js=js[js<len(arr)-i]
    return np.min((arr[i+js]+D-now-extra)/(js+1-start)) if len(js) else np.inf, q
def run(arr,D,menu,alpha,top,refine=None,theta=None,coin=None,seed=7):
    """menu: list of (name,cost,ap) sorted by cost; refine: dict base-> (t_r, ap_ref) for late binding."""
    r=np.random.default_rng(seed); menu=[m for m in menu if m[1]<=top]; tmin=menu[0][1]
    free=0.0; tot=0.0; miss=0; nref=0; nbase=0; i=0; N=len(arr)
    while i<N:
        now=max(free,arr[i])
        if now+tmin>arr[i]+D: miss+=1; i+=1; continue
        b,_=budget(arr,i,now,D)
        ok=[m for m in menu if m[1]<=alpha*b and now+m[1]<=arr[i]+D]
        if not ok: ok=[m for m in menu if now+m[1]<=arr[i]+D][:1]
        m=ok[-1]; now+=m[1]; ap=m[2]
        if refine and m[0] in refine:
            nbase+=1; tr,apr=refine[m[0]]
            if coin is None:
                bb,_=budget(arr,i,now,D,extra=tr,start=1)   # waiting frames after a refine
                go = now+tr<=arr[i]+D and bb>=theta
            else: go = r.random()<coin
            if go:
                now+=tr
                if now<=arr[i]+D: ap=apr; nref+=1
        tot+=ap; free=now; i+=1
    return tot/N, miss/N, (nref/max(nbase,1))
def pareto(menu):
    menu=sorted(menu,key=lambda m:m[1]); out=[]
    for m in menu:
        if not out or m[2]>out[-1][2]: out.append(m)
    return out
ALPHAS=[0.75,0.9,1.0]; THETAS=[4.3,4.9,5.4]
def best(arr,D,menu,refine=None,late=False):
    res=None
    tops=[m[1] for m in menu if m[1]>=4.88]
    for alpha,top in itertools.product(ALPHAS,tops):
        for th in (THETAS if late else [None]):
            ap,mf,sh=run(arr,D,menu,alpha,top,refine if late else None,th)
            if res is None or ap>res[0]: res=(ap,mf,sh,alpha,top,th)
    return res
scen=[(5.6,20.0,None),(6.0,12.0,None),(6.0,20.0,None),(7.0,20.0,None),(8.0,20.0,None),(7.0,15.0,3.0),(8.0,20.0,4.0),(6.5,33.0,None),(10.0,33.0,None)]
for phi in [1.0,0.6,0.4]:
  for tr_name,tr in [('mid',0.91),('hi',1.23)]:
    REF={b:(tr, A[b]+phi*(CEIL[b]-A[b])) for b in CEIL}
    early_menu=pareto(DENSE+[(b+'+R',T[b]+tr,REF[b][1]) for b in REF])
    print(f'\n### phi={phi} refiner {tr_name} {tr:.2f} ms; early menu: '+' '.join(f'{m[0]}({m[1]:.2f},{m[2]:.4f})' for m in early_menu),flush=True)
    print('scenario | dense AP miss% (alpha,top) | early AP miss% | late AP miss% refine-share (alpha,top,theta) | null AP miss% | late-dense | late-early | late-null')
    for mean_ia,D,burst in scen:
        arr=arrivals(mean_ia,burst,1)
        d=best(arr,D,pareto(DENSE)); e=best(arr,D,early_menu); l=best(arr,D,pareto(DENSE),REF,late=True)
        nu=run(arr,D,pareto(DENSE),l[3],l[4],REF,None,coin=l[2])
        tag=f'ia {mean_ia}{"x"+str(burst) if burst else ""} D {D:.0f}'
        print(f'{tag} | {d[0]:.4f} {d[1]*100:.1f}% ({d[3]},{d[4]}) | {e[0]:.4f} {e[1]*100:.1f}% | {l[0]:.4f} {l[1]*100:.1f}% {l[2]:.2f} ({l[3]},{l[4]},{l[5]}) | {nu[0]:.4f} {nu[1]*100:.1f}% | {l[0]-d[0]:+.4f} | {l[0]-e[0]:+.4f} | {l[0]-nu[0]:+.4f}',flush=True)
