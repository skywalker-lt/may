"""Final table for the revised construct (CN-stem): route = native-resolution noise floor -> gn3_k21; thumbnail contrast
posterior > 0.9 -> ct3_k21; else public. Realised route simulated with the measured router rates (s8, s10): noise recall at
severity 3 1.00, clean->noise 0.0025; contrast recall 0.97, clean->contrast 0.0025; blur/brightness->either 0.00-0.005.
Dense comparators: public (D0), pooled_k21 and gnct_k21 (moment-matched pooled stem sets, D1), single-image alpha-BN (D3)."""
import sys, json, numpy as np
sys.argv = ['x']; exec(open('/data/tmp/ds-yolo/seminar6/work/agent4/s4_score.py').read().split("conds = ['clean'")[0])
allR = {}
for fpath in ['matrix_dets.json', 'matrix2_dets.json', 'matrix3_dets.json', 'matrix4_dets.json']:
    try: allR.update(json.load(open(f'{OUT}/{fpath}')))
    except FileNotFoundError: print('missing', fpath)
recs = {k: rec_of(v) for k, v in allR.items() if v}; AP = {k: full_ap(v)[0] for k, v in allR.items() if v}
conds = ['clean', 'gn3', 'db3', 'ct3', 'br3']; cor = conds[1:]
rng = np.random.default_rng(0); boots = [rng.integers(0, n, n) for _ in range(200)]
route_rates = {'clean': {'gn3_k21': 0.0025, 'ct3_k21': 0.0025}, 'gn3': {'gn3_k21': 1.0}, 'db3': {}, 'ct3': {'ct3_k21': 0.97}, 'br3': {'gn3_k21': 0.0, 'ct3_k21': 0.005}}
def have(c, s): return f'{c}__{s}' in recs
def routed_assign(c, r):
    names = ['public'] + [s for s in route_rates[c] if have(c, s)]
    p = [1 - sum(route_rates[c][s] for s in names[1:])] + [route_rates[c][s] for s in names[1:]]
    return names, r.choice(len(names), n, p=p)
# expected route AP (average over 20 assignment draws) and paired bootstrap of route minus comparator
rr = np.random.default_rng(5); draws = {c: [routed_assign(c, rr) for _ in range(20)] for c in conds}
def route_ap(c, idx=None):
    return np.mean([ap_assign([recs[f'{c}__{s}'] for s in nm], a, idx) for nm, a in draws[c][:(20 if idx is None else 3)]])
comps = ['public', 'pooled_k21', 'gnct_k21', 'abn0.2', 'abn0.5']
print('n=%d images; AP per condition (severity 3):' % n)
print('%-8s %8s ' % ('cond', 'route') + ' '.join('%10s' % c for c in comps))
RT = {}
for c in conds:
    RT[c] = route_ap(c); print('%-8s %8.4f ' % (c, RT[c]) + ' '.join('%10s' % ('%.4f' % AP[f'{c}__{s}'] if have(c, s) else '-') for s in comps))
mpc = lambda f: np.mean([f(c) for c in cor])
print('mPC(4)   %8.4f ' % mpc(lambda c: RT[c]) + ' '.join('%10s' % ('%.4f' % mpc(lambda c: AP[f'{c}__{s}']) if all(have(c, s) for c in cor) else '-') for s in comps))
print('\nPaired bootstrap (200 draws over images), route minus comparator: mPC over the 4 corruptions and clean AP')
z = np.zeros(n, int)
RCB = [{c: route_ap(c, b) for c in conds} for b in boots]
for s in comps:
    if not all(have(c, s) for c in conds): continue
    dm, dc = [], []
    for b, rc in zip(boots, RCB):
        dc.append(rc['clean'] - ap_assign([recs[f'clean__{s}']], z, b))
        dm.append(np.mean([rc[c] - ap_assign([recs[f'{c}__{s}']], z, b) for c in cor]))
    print('vs %-11s mPC %+.4f (sd %.4f) | clean %+.4f (sd %.4f)' % (s, mpc(lambda c: RT[c]) - mpc(lambda c: AP[f'{c}__{s}']), np.std(dm), RT['clean'] - AP[f'clean__{s}'], np.std(dc)))
