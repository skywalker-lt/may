import re,sys,collections
def agg(p):
    d=collections.OrderedDict(); tot=None
    for line in open(p):
        if line.startswith('#'):
            tot=float(re.search(r'total ([\d.]+)',line).group(1)); continue
        ms,name=line.strip().split(None,1); ms=float(ms)
        m=re.search(r'/model\.(\d+)/',name)
        k=int(m.group(1)) if m else -1
        d[k]=d.get(k,0)+ms
    return tot,d
for p in sys.argv[1:]:
    tot,d=agg(p); print(p.split('/')[-1],'total',tot)
    print('  '+' '.join(f'{k}:{v:.3f}' for k,v in sorted(d.items())))
    stem=sum(v for k,v in d.items() if 0<=k<=5); back=sum(v for k,v in d.items() if 6<=k<=10); neck=sum(v for k,v in d.items() if 11<=k<=22); head=d.get(23,0); other=d.get(-1,0)
    print(f'  stem0-5 {stem:.3f} backbone6-10 {back:.3f} neck11-22 {neck:.3f} head23 {head:.3f} unattributed {other:.3f}')
