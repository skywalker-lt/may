# Attribute TensorRT per-layer profile time of YOLO26 to pyramid branches.
import sys, re, collections
def grp(n):
    m = re.search(r'/model\.(\d+)/', n)
    if 'one2one_cv' in n:
        lv = re.search(r'one2one_cv[23]\.(\d)', n).group(1)
        return {'0':'P3 head','1':'P4 head','2':'P5 head'}[lv]
    if n.startswith('__myl') or 'Topk' in n or 'model.23' in n: return 'post (decode/TopK)'
    if m is None:
        if 'Input Tensor 0 to /model.0' in n: return 'input reformat'
        return 'other'
    i = int(m.group(1))
    if i <= 4: return 'backbone 0-4 (to P3)'
    if i <= 6: return 'backbone 5-6 (to P4)'
    if i <= 10: return 'backbone 7-10 (to P5)'
    if i <= 13: return 'neck 11-13 (P4 top-down)'
    if i <= 17: return 'neck 14-17 (P3 path)'
    if i <= 19: return 'neck 18-19 (P4 out)'
    if i <= 22: return 'neck 20-22 (P5 out)'
    return 'post (decode/TopK)'
for p in sys.argv[1:]:
    c = collections.defaultdict(float); tot = 0
    for l in open(p):
        if l.startswith('#'): hdr = l.strip(); continue
        t, name = l.strip().split(None, 1); t = float(t); c[grp(name)] += t; tot += t
    print(p.split('/')[-1], hdr, 'sum', round(tot, 3))
    for k in sorted(c, key=lambda k: -c[k]): print(f'  {k:28} {c[k]:.3f}  {100*c[k]/tot:5.1f}%')
