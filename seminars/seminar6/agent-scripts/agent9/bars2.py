"""Bars for a classification-only re-scoring expert on L@512 (T4, est. cost +0.28 / +0.50 / +0.77 ms) and the fraction f of
the X@640 re-score-only bound (+0.0178 over L@512, rescore_l512.log) each bar needs. Reuses bars.py's envelope functions."""
import importlib.util, io, contextlib
spec=importlib.util.spec_from_file_location('b','/data/tmp/ds-yolo/seminar6/work/agent9/bars.py'); b=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(b)
tb,ab=b.pts['L@512']; R=0.5440-ab
for dT in (0.28,0.50,0.77,0.81):
    t=tb+dT; e=[b.interp(t),b.hull(t),b.hull(t,0.26)]; lo,hi=min(e)+0.003,max(e)+0.003; f=0.5261+0.0102*(t-5.36)+0.003
    print(f'L@512 +{dT:.2f} -> {t:.2f} ms: envelope+0.003 {lo:.4f}-{hi:.4f} (needs +{lo-ab:.4f} to +{hi-ab:.4f}, f {(lo-ab)/R:.2f}-{(hi-ab)/R:.2f}); packet bar {f:.4f} (f {(f-ab)/R:.2f})')
