"""Round 3: the dense envelope over (model, input scale) on the T4 (unset basis, est. where unmeasured) under three cost models
for L at reduced scale, the L@544 interpolation, and the best count-routed menus against it. CPU, one thread."""
import os, sys, itertools, numpy as np
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7")
F = lambda L: 0.5261 + 0.0102 * (L - 5.36); R = 0.18
AP = {"S@640": 0.4795, "M@448": 0.4937, "M@512": 0.5063, "M@576": 0.5203, "M@608": 0.5234, "M@640": 0.5261, "M@768": 0.5196,
      "L@448": 0.5117, "L@512": 0.5262, "L@576": 0.5368, "L@640": 0.5417}
MEAS = {"S@640": 2.75, "M@512": 3.78, "M@640": 5.36, "M@768": 6.97, "L@640": 6.89}
def Mc(s): p = (s / 640) ** 2; return {512: 3.78, 640: 5.36, 768: 6.97}.get(s, 3.78 + (p - 0.64) * 4.389)
LM = {"A ratio (L scales like M)": lambda s: 6.89 * Mc(s) / 5.36,
      "B pure pixels": lambda s: 6.89 * (s / 640) ** 2,
      "C L fixed term 2x": lambda s: 2.50 + 4.39 * (s / 640) ** 2}
def cost(k, lm):
    if k in MEAS: return MEAS[k]
    f, s = k[0], int(k[2:]); return Mc(s) if f == "M" else LM[lm](s)
def hull_at(pts, L):
    best = max([a for c, a in pts if c <= L + 1e-9] + [-1])
    for (c0, a0), (c1, a1) in itertools.combinations(sorted(pts), 2):
        if c0 <= L <= c1 and c1 > c0: best = max(best, a0 + (a1 - a0) * (L - c0) / (c1 - c0))
    return best
# L@544 interpolation in ln s
ls = np.log([448, 512, 576, 640]); la = np.array([AP["L@448"], AP["L@512"], AP["L@576"], AP["L@640"]])
lin = AP["L@512"] + (AP["L@576"] - AP["L@512"]) * (np.log(544 / 512) / np.log(576 / 512))
cub = np.polyval(np.polyfit(ls, la, 3), np.log(544)); quad = np.polyval(np.polyfit(ls, la, 2), np.log(544))
print(f"L@544 interp (est.): linear-ln {lin:.4f}  quadratic-ln {quad:.4f}  cubic-ln {cub:.4f}")
for lm in LM:
    print(f"\n== cost model {lm}")
    for k in sorted(AP, key=lambda k: cost(k, lm)):
        c = cost(k, lm); print(f"  {k:6s} T4 {c:5.2f} ms {'meas.' if k in MEAS else 'est.'}  AP {AP[k]:.4f}  AP-(F+0.003) {AP[k]-F(c)-0.003:+.4f}")
    c544 = LM[lm](544); print(f"  L@544  T4 {c544:5.2f} ms est.   AP {lin:.4f}-{max(quad,cub):.4f} interp.  AP-(F+0.003) {lin-F(c544)-0.003:+.4f}")
    pts = [(cost(k, lm), AP[k]) for k in AP]
    for T in (4.0, 4.5, 5.05, 5.36, 5.8, 6.3, 6.89):
        print(f"  envelope at {T:.2f} ms: {hull_at(pts, T):.4f}  (F+0.003 {F(T)+0.003:.4f}; envelope - F {hull_at(pts, T)-F(T):+.4f})")
if len(sys.argv) > 1 and sys.argv[1] == "menus":
    from headroom import I, cnt_pred_all, cnt_pred_small, n_all
    from mixlib import mix_ap
    N = {"M@448": "dumpml_yolo26m_448_coco", "M@512": "dumpml_yolo26m_512_coco", "M@640": "dumpml_yolo26m_coco", "L@448": "dumpml_yolo26l_448_coco",
         "L@512": "dumpml_yolo26l_512_coco", "L@576": "dumpml_yolo26l_576_coco", "L@640": "dump_yolo26l_coco"}
    def ranks(score): r = np.empty(I, int); r[np.argsort(score + 1e-9 * np.arange(I))] = np.arange(I); return r
    RP = ranks(cnt_pred_all + 1e-6 * cnt_pred_small); RG = ranks(n_all + 1e-3 * np.random.RandomState(0).rand(I))
    def assign(r, sh): return np.searchsorted(np.cumsum(sh)[:-1] * I, r, side="right")
    def shares(keys, q, T, lm):
        c = [cost(k, lm) for k in keys]
        if len(keys) == 2:
            x = (c[1] + R - T) / (c[1] - c[0]); return [x, 1 - x] if 0 < x < 1 else None
        qe = (T - R - c[1] + q * (c[1] - c[0])) / (c[2] - c[1]); return [q, 1 - q - qe, qe] if 0 < qe < 1 - q else None
    MENUS = [(("M@512", "M@640", "L@640"), 0.5), (("L@448", "L@512", "L@640"), 0.33), (("M@448", "L@512", "L@640"), 0.2),
             (("M@448", "L@576"), None), (("L@512", "L@576"), None), (("L@448", "L@576"), None)]
    for lm in LM:
        pts = [(cost(k, lm), AP[k]) for k in AP]
        for T in ((5.36,) if lm[0] != "A" else (4.6, 5.36, 6.0)):
            for keys, q in MENUS:
                sh = shares(keys, q, T, lm)
                if sh is None: continue
                names = [N[k] for k in keys]; ap = mix_ap(names, assign(RP, sh)); e = hull_at(pts, T)
                extra = ""
                if lm[0] == "A" and T == 5.36 and keys in (("L@448", "L@512", "L@640"), ("M@448", "L@512", "L@640")):
                    g = mix_ap(names, assign(RG, sh)); a = assign(RP, sh); b = a.copy(); b[a < len(keys) - 1] = len(keys) - 1
                    extra = f" | GT-count rule {g:.4f} (-env-0.003 {g-e-0.003:+.4f}) | cheap rungs scored as top member (spec. bound) {mix_ap(names, b):.4f}"
                print(f"[{lm[0]}] T {T:.2f} {'/'.join(keys):20s} shares {np.round(sh,3)} AP {ap:.4f} | -bar {ap-F(T)-0.003:+.4f} | env {e:.4f} -env-0.003 {ap-e-0.003:+.4f}{extra}", flush=True)
