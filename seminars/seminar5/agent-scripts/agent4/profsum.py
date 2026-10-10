import sys,re,collections
for f in sys.argv[1:]:
    tot=0; d=collections.defaultdict(float); attn=collections.defaultdict(float); lines=[]
    for l in open(f):
        if l.startswith('#'): print(l.strip()); continue
        p=l.strip().split(None,1)
        if len(p)<2: continue
        try: t=float(p[0])
        except: continue
        tot+=t
        m=re.findall(r'/model\.(\d+)/',p[1])
        key=m[0] if m else 'other'
        d[key]+=t
        if 'attn' in p[1] or 'mha' in p[1].lower() or 'Softmax' in p[1] or 'MatMul' in p[1]: attn[key]+=t; lines.append((t,p[1][:150]))
    print(f, 'sum',round(tot,4))
    print({k:round(v,4) for k,v in sorted(d.items(),key=lambda x:(len(x[0]),x[0]))})
    print('attn-ish',{k:round(v,4) for k,v in attn.items()})
