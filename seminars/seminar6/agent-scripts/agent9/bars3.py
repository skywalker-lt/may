"""Round 2 bars. Square envelope (measured T4 points, interpolation / hull / hull shifted by the +0.26 ms branch penalty)
for L@544 and L@512 bases with the tile re-scorer (+0.28/0.50/0.77 ms est.) and agent 1's query unit (+0.24/0.35/0.58 est.).
Rect comparator: agent 5's rect-minus-square hull offsets (est.), interpolated: 4.6 +0.0136, 4.9 +0.0123, 5.3 +0.0087,
5.6 +0.0073, 6.0 +0.0050; rect L@544 base avg 4.51 ms est."""
import importlib.util, io, contextlib, numpy as np
spec=importlib.util.spec_from_file_location('b','/data/tmp/ds-yolo/seminar6/work/agent9/bars.py'); b=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(b)
rect_t=[4.3,4.6,4.9,5.3,5.6,6.0]; rect_d=[0.0142,0.0136,0.0123,0.0087,0.0073,0.0050]
def env(t): e=[b.interp(t),b.hull(t),b.hull(t,0.26)]; return min(e),max(e)
print('square comparator (T4 measured points):')
for base,tb,ab in (('L@544',5.32,0.5329),('L@512',4.88,0.5262)):
    for unit,costs in (('tile',(0.28,0.50,0.77)),('query',(0.24,0.35,0.58))):
        for dT in costs:
            t=tb+dT; lo,hi=env(t); print(f'  {base} {unit:5s} +{dT:.2f} -> {t:.2f} ms: envelope+0.003 {lo+0.003:.4f}-{hi+0.003:.4f}; needed over base +{lo+0.003-ab:.4f} to +{hi+0.003-ab:.4f}')
print('rect comparator: agent 5 rect hull points (est.) 4.70 ms 0.5355, 5.01 ms 0.5384, 5.42 ms 0.5415; rect L@544 base avg 4.51 ms est.')
rh_t=[4.70,5.01,5.42]; rh_a=[0.5355,0.5384,0.5415]
for unit,costs in (('tile',(0.28,0.50,0.77)),('query',(0.24,0.35,0.58))):
    for dT in costs:
        t=4.51+dT; h=np.interp(t,rh_t,rh_a); print(f'  rect L@544 {unit:5s} +{dT:.2f} -> {t:.2f} ms avg: rect hull {h:.4f}, bar {h+0.003:.4f}; needed over 0.5329 +{h+0.003-0.5329:.4f}')
t=5.38; h=np.interp(t,rh_t,rh_a); print(f'  square L@512 tile +0.50 -> 5.38 ms against the rect hull: bar {h+0.003:.4f}; needed over 0.5262 +{h+0.003-0.5262:.4f}')
