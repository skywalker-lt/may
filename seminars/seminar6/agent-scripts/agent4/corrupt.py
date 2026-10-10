"""COCO-C style corruptions, re-implemented from the imagecorruptions package formulas (the package is not installed).
RGB float images in [0,255]. One representative per family: gaussian_noise (noise), defocus_blur (blur),
brightness (weather-side, the only weather corruption without a plasma fractal), contrast (digital)."""
import numpy as np, cv2

def disk(radius, alias_blur=0.1):
    if radius <= 8: L = np.arange(-8, 8 + 1); ksize = (3, 3)
    else: L = np.arange(-radius, radius + 1); ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    d = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=np.float32); d /= d.sum()
    return cv2.GaussianBlur(d, ksize=ksize, sigmaX=alias_blur)

def gaussian_noise(x, s, rng):
    c = [.08, .12, 0.18, 0.26, 0.38][s - 1]
    x = x / 255.
    return np.clip(x + rng.normal(size=x.shape, scale=c), 0, 1) * 255

def defocus_blur(x, s, rng=None):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][s - 1]
    x = (x / 255.).astype(np.float32); k = disk(radius=c[0], alias_blur=c[1])
    ch = [cv2.filter2D(x[:, :, d], -1, k) for d in range(3)]
    return np.clip(np.stack(ch, -1), 0, 1) * 255

def contrast(x, s, rng=None):
    c = [0.4, .3, .2, .1, .05][s - 1]
    x = x / 255.; m = x.mean(axis=(0, 1), keepdims=True)
    return np.clip((x - m) * c + m, 0, 1) * 255

def brightness(x, s, rng=None):
    c = [.1, .2, .3, .4, .5][s - 1]
    x = (x / 255.).astype(np.float32)
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)  # float32: H in [0,360], S,V in [0,1]
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + c, 0, 1)
    return np.clip(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), 0, 1) * 255

FUN = {"gn": gaussian_noise, "db": defocus_blur, "ct": contrast, "br": brightness}

def apply(im_bgr_u8, cond, rng):
    """cond: 'clean' or '<name><severity>' e.g. 'gn3'. Returns BGR uint8."""
    if cond == "clean": return im_bgr_u8
    name, s = cond[:2], int(cond[2])
    x = im_bgr_u8[:, :, ::-1].astype(np.float64)
    y = FUN[name](x, s, rng)
    return np.ascontiguousarray(np.asarray(y, np.float64).round().clip(0, 255).astype(np.uint8)[:, :, ::-1])
