import re,sys,collections
f=sys.argv[1]; agg=collections.defaultdict(float); tot=0; topk=0
for line in open(f):
    if line.startswith('#'): print(line.strip()); continue
    p=line.split(None,1)
    if len(p)<2: continue
    t=float(p[0]); name=p[1]; tot+=t
    m=re.search(r'/model\.(\d+)/',name)
    key='other'
    if m:
        L=int(m.group(1)); key=f'L{L:02d}'
        if L==23:
            h=re.search(r'(one2one_cv[23])\.(\d)',name)
            key='head_'+(h.group(1)+'_P'+str(3+int(h.group(2))) if h else 'post')
    if 'TopK' in name or 'topk' in name.lower(): topk+=t
    agg[key]+=t
for k in sorted(agg): print(f'{k:20s} {agg[k]:.4f}')
print('sum',round(tot,4),'topk-ish',round(topk,4))
