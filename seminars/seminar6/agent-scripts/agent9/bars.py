"""Bars on the measured T4 envelope (T4-envelope.md) for a static tile expert added to a base point.
Expert cost on the T4: agent 5's translation of the H200 profile, est. +0.49 / +0.81 / +1.13 ms (low / mid / high).
Two readings of the envelope: (i) the moderator's interpolation through the dense points; (ii) the upper hull, where a
chord between two dense points is a random per-image route (which on the T4 pays the +0.26 ms Resize-fed branch penalty,
so the chord is also shown shifted right by 0.26 ms). Every latency here is T4 and est. unless it is a measured dense point."""
import numpy as np
pts={'L@512':(4.88,0.5262),'L@544':(5.32,0.5329),'L@576':(6.25,0.5368),'L@640':(6.89,0.5417),'M@640':(5.34,0.5261)}
L=[pts[k] for k in ('L@512','L@544','L@576','L@640')]
def interp(t):
    for (t0,a0),(t1,a1) in zip(L[:-1],L[1:]):
        if t0<=t<=t1: return a0+(a1-a0)*(t-t0)/(t1-t0)
def hull(t,pen=0.0):
    t=t-pen; best=0
    for i in range(len(L)):
        for j in range(i+1,len(L)):
            (t0,a0),(t1,a1)=L[i],L[j]
            if t0<=t<=t1: best=max(best,a0+(a1-a0)*(t-t0)/(t1-t0))
    return best
print('base      dT    T4 ms   env(interp)  env(hull)  env(hull,+0.26 pen)   bar range (+0.003)   needed over base')
for base in ('M@640','L@512'):
    tb,ab=pts[base]
    for dT in (0.49,0.81,1.13):
        t=tb+dT; e=[interp(t),hull(t),hull(t,0.26)]; lo,hi=min(e)+0.003,max(e)+0.003
        print(f'{base:8s} +{dT:.2f}  {t:5.2f}   {e[0]:.4f}       {e[1]:.4f}     {e[2]:.4f}              {lo:.4f}-{hi:.4f}      +{lo-ab:.4f} to +{hi-ab:.4f}')
print()
print('front F(L)=0.5261+0.0102(L-5.36) packet bar (+0.003) for reference:')
for base in ('M@640','L@512'):
    tb,ab=pts[base]
    for dT in (0.49,0.81,1.13):
        t=tb+dT; f=0.5261+0.0102*(t-5.36)+0.003; print(f'  {base} +{dT:.2f} -> {t:.2f} ms: packet bar {f:.4f} (+{f-ab:.4f} over base)')
