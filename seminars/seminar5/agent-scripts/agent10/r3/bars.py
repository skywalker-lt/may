# T4 bars (unset basis) for add-on constructs: F+0.003 and dense envelope+0.003; envelope points as agent 7's menus.py (L-at-scale, M@448/576/608 costs est.)
import itertools
pts={'M@448':(3.12,.4937),'M@512':(3.78,.5063),'L@448':(4.01,.5117),'M@576':(4.53,.5203),'L@512':(4.86,.5262),'M@608':(4.93,.5234),'M@640':(5.36,.5261),'L@576':(5.82,.5368),'L@640':(6.89,.5417)}
F=lambda L:.5261+.0102*(L-5.36)
def env(L,scale=1.0):
    h=[(c*(scale if k.startswith('L') and k!='L@640' else 1),a) for k,(c,a) in pts.items()]
    b=max(a for c,a in h if c<=L)
    for (c0,a0),(c1,a1) in itertools.combinations(sorted(h),2):
        if c0<=L<=c1 and c1>c0: b=max(b,a0+(a1-a0)*(L-c0)/(c1-c0))
    return b
print("env - F by latency (L-at-scale cost x1.00 / x0.97 / x1.03):")
for L in (4.6,5.0,5.2,5.36,5.6,5.8,6.0,6.2,6.5,6.89):
    print(f"  {L:.2f} ms  F {F(L):.4f}  env {env(L):.4f} ({env(L,.97):.4f}/{env(L,1.03):.4f})  env-F {env(L)-F(L):+.4f}")
print("TCR on public M (5.36) + d ms: needed gain over M for (a) F+0.003 and (c) env+0.003")
for d in (0.65,0.85,1.10):
    L=5.36+d; print(f"  +{d:.2f} -> {L:.2f} ms: (a) +{F(L)+.003-.5261:.4f}  (c) +{env(L)+.003-.5261:.4f}")
print("same expert on L@576 (5.82 est.) + d: needed gain over L@576 (0.5368)")
for d in (0.65,0.85,1.10):
    L=5.82+d; print(f"  +{d:.2f} -> {L:.2f} ms: (a) +{F(L)+.003-.5368:.4f}  (c, env to L@640 then flat) +{(env(min(L,6.89))+.003-.5368) if L<=6.89 else float('nan'):.4f}")
print("late-branch line at dense-in-If latency, both sides given the If effect: (a) 0.5291; (c) env(5.36)+0.003 =",round(env(5.36)+.003,4))
