"""T4 cost model for rect (native-aspect) engines, est. from the measured square series (T4-envelope.md).
All outputs are est.; nothing rect-shaped has been timed on the T4."""
import numpy as np

# measured T4 unset-buffer medians (ms) of static square engines, from T4-envelope.md / t4_r1.log / T4-baselines.md
SQ = {
    'l': {448: 4.28, 480: 4.61, 512: 4.88, 544: 5.32, 576: 6.25, 640: 6.89},
    'm': {448: 3.33, 512: 3.78, 576: 4.93, 608: 5.20, 640: 5.34, 768: 6.97},
}

def kpix(w, h):
    return w * h / 1000.0

def cost_interp(model, w, h):
    """central: piecewise-linear in pixel count through the measured square points."""
    pts = sorted((kpix(s, s), ms) for s, ms in SQ[model].items())
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    return float(np.interp(kpix(w, h), xs, ys))

def cost_linfit(model, w, h):
    """low/high band: least-squares line ms = a + b*kpix through the measured points below 640 (L) or all (M)."""
    pts = sorted((kpix(s, s), ms) for s, ms in SQ[model].items() if s <= 640)
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    b, a = np.polyfit(xs, ys, 1)
    return float(a + b * kpix(w, h))

def rect_shape(long_side, w_img, h_img, stride=32):
    """ultralytics-style rect: long side = long_side, short side scaled and padded up to a stride multiple."""
    lo, hi = min(w_img, h_img), max(w_img, h_img)
    short = int(np.ceil(long_side * lo / hi / stride) * stride)
    return (long_side, short) if w_img >= h_img else (short, long_side)

if __name__ == '__main__':
    for model in ('l', 'm'):
        print(f'model {model}: square check (interp reproduces the measured points by construction); linfit band:')
        for s, ms in sorted(SQ[model].items()):
            print(f'  {s}x{s} {kpix(s,s):6.1f} kpix meas {ms:.2f}  linfit {cost_linfit(model,s,s):.2f}')
        print('  rect shapes (est.):')
        for (w, h) in [(640, 480), (640, 448), (640, 416), (640, 384), (608, 448), (576, 448), (576, 416), (544, 416), (544, 384), (512, 384), (512, 352), (512, 320), (448, 352), (448, 320), (448, 288), (736, 544), (704, 480)]:
            print(f'  {w}x{h} {kpix(w,h):6.1f} kpix  interp {cost_interp(model,w,h):.2f}  linfit {cost_linfit(model,w,h):.2f}')
