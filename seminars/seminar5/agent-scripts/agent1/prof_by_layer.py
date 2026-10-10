import re,sys,collections
f=sys.argv[1]; agg=collections.defaultdict(float); tot=0
for line in open(f):
    if line.startswith('#'): print(line.strip()); continue
    t,name=line.strip().split(None,1); t=float(t); tot+=t
    m=re.search(r'/model\.(\d+)/',name)
    k=int(m.group(1)) if m else -1
    if k==23:
        m2=re.search(r'one2one_cv(\d)\.(\d)',name)
        k=f"23.cv{m2.group(1)}.{m2.group(2)}" if m2 else "23.other"
    agg[str(k)]+=t
for k in sorted(agg,key=lambda s:(len(s.split('.')[0]),s)): print(f"{k:>12} {agg[k]:.4f}")
print("sum",round(tot,4))
