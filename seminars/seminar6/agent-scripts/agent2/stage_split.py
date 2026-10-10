import re,sys,collections
for f in sys.argv[1:]:
    tot=0; by=collections.OrderedDict(); other=0
    for line in open(f):
        m=re.match(r'\s*([\d.]+)\s+(.*)',line)
        if not m: continue
        t=float(m.group(1)); name=m.group(2); tot+=t
        ms=re.findall(r'/model\.(\d+)/',name)
        if ms:
            k=int(ms[0]); by[k]=by.get(k,0)+t
        else: other+=t
    print(f, 'total', round(tot,3), 'unattributed', round(other,3))
    for k in sorted(by): print('  layer',k, round(by[k],3))
    def S(a,b): return sum(v for k,v in by.items() if a<=k<=b)
    print('  stem+P2+P3 (0-4)', round(S(0,4),3), ' P4 (5-6)', round(S(5,6),3), ' P5 (7-8)', round(S(7,8),3), ' SPPF+PSA (9-10)', round(S(9,10),3), ' neck (11-22)', round(S(11,22),3), ' head (23)', round(S(23,23),3))
