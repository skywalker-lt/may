"""Ladder B/A/C with C' (P2-leaf proxy: 768 detections under 16 px only) instead of the full M@768 proxy."""
exec(open('levels_vs_res.py').read().split("print('--- 1-bit")[0])
Cl = load('cleaf_proxy')
print('C\' everywhere (dense) AP', mix_ap([Cl], np.zeros(I, int))[0])
for sC in (0.2, 0.3):
    rep(f'C\' only (A / C\'), C share {sC}, +1.3 ms', [Mf, Cl], np.where(route3(rm, 0, sC) == 2, 1, 0), [5.36, 6.66])
for nm, Bc in (('B tau16', B16), ('B tau32', B32)):
    for cC in (1.0, 1.3, 1.6):
        rep(f'level: {nm} / A / C\'(P2-leaf proxy, +{cC} ms), m640', [Bc, Mf, Cl], route3(rm, 0.49, 0.20), [4.36, 5.36, 5.36 + cC])
