"""Split the H200 TCR stub profiles into M / router / dispatch / expert / combine / final TopK, by the order of the
kernels in the last Myelin graph (head decode -> router -> dispatch -> expert -> combine -> final TopK)."""
import re,sys,collections
def split(f, kind):
    st=collections.defaultdict(float); n=collections.Counter(); tot=0
    rows=[]
    for line in open(f):
        if line.startswith('#'): continue
        p=line.split(None,1); t=float(p[0]); name=p[1].strip(); tot+=t
        m=re.search(r'_myl(229|234)_(\d+)$',name)
        if not m: st['M (layers, head convs, M attention)']+=t; n['M (layers, head convs, M attention)']+=(t>0); continue
        i=int(m.group(2)); rows.append((i,t,name))
    rows.sort()
    if kind=='gather':
        cut={'head decode (in M too)':range(0,2),'router':range(2,7),'dispatch':range(7,8),'expert':range(8,45),
             'combine':range(45,49),'final TopK/gather (in M too, over 13,520 anchors)':range(49,53)}
    else:
        cut={'head decode (in M too)':range(3,5),'router':range(5,10),'dispatch':range(10,29),'expert':range(29,66),
             'combine':range(66,73),'final TopK/gather (in M too, over 13,520 anchors)':range(73,77)}
    for i,t,name in rows:
        for k,r in cut.items():
            if i in r: st[k]+=t; n[k]+=(t>0)
    return st,n,tot
for f,kind,med in [(sys.argv[1],'gather',1.490),(sys.argv[2],'onehot',1.637)]:
    st,n,tot=split(f,kind); print(f'== {kind}: profiled total {tot:.4f} ms, median {med} ms (profile/median {tot/med:.3f})')
    for k,v in st.items(): print(f'   {k:55s} {v:.4f} ms profiled  {n[k]:3d} kernels  share {v/tot:.3f}')
    add=st['router']+st['dispatch']+st['expert']+st['combine']
    print(f'   router+dispatch+expert+combine profiled {add:.4f}; median delta vs tcr_m_base 1.281 = {med-1.281:.3f} ms; ratio {add/(med-1.281):.2f}')
