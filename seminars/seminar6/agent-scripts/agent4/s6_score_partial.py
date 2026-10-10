import sys, json, numpy as np
sys.argv = ['x']; exec(open('/data/tmp/ds-yolo/seminar6/work/agent4/s4_score.py').read().split("conds = ['clean'")[0])
R2 = json.load(open(f'{OUT}/matrix2_dets.json')); R1 = R
recs = {}; AP = {}
for k, v in list(R1.items()) + list(R2.items()):
    recs[k] = rec_of(v); AP[k] = full_ap(v)[0]
rng = np.random.default_rng(0); boots = [rng.integers(0, n, n) for _ in range(200)]; z = np.zeros(n, int)
def diff(k1, k2):
    d = np.array([ap_assign([recs[k1]], z, b) - ap_assign([recs[k2]], z, b) for b in boots]); return AP[k1] - AP[k2], d.std()
print('n=%d. Partial BN experts (family stats in the first k BN layers, public elsewhere), minus public:' % n)
for c in ['gn3', 'db3', 'ct3', 'br3']:
    print('%s: public %.4f | k2 %+.4f (sd %.4f) | k21 %+.4f (sd %.4f) | full %+.4f' % ((c, AP[f'{c}__public']) + diff(f'{c}__{c}_k2', f'{c}__public') + diff(f'{c}__{c}_k21', f'{c}__public') + (AP[f'{c}__{c}'] - AP[f'{c}__public'],)))
print('clean misroute tax (minus public %.4f): ' % AP['clean__public'] + ' '.join('%s %+.4f (sd %.4f)' % ((s,) + diff(f'clean__{s}', 'clean__public')) for s in ['cleanrecal_k21'] + [f'{f}_k{k}' for f in ['gn3', 'db3', 'ct3', 'br3'] for k in (2, 21)]))
for k in (2, 21):
    print('mPC (4 conditions, severity 3): public %.4f | oracle family k%d %.4f | best-of(public, k%d) per condition %.4f' % (np.mean([AP[f'{c}__public'] for c in ['gn3', 'db3', 'ct3', 'br3']]), k, np.mean([AP[f'{c}__{c}_k{k}'] for c in ['gn3', 'db3', 'ct3', 'br3']]), k, np.mean([max(AP[f'{c}__public'], AP[f'{c}__{c}_k{k}']) for c in ['gn3', 'db3', 'ct3', 'br3']])))
