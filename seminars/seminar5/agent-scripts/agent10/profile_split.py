import re,sys
def split(path):
    tot=0;cats={}
    stem=0;tail=0;head=0
    for line in open(path):
        if line.startswith('#'): continue
        m=re.match(r'\s*([\d.]+)\s+(.*)',line)
        if not m: continue
        t=float(m.group(1)); n=m.group(2); tot+=t
        if 'Reformatting' in n: c='reformat'
        elif 'MatMul' in n or 'gemm' in n.lower() or 'mha' in n.lower() or 'Softmax' in n: c='attention/matmul'
        elif '/conv/Conv' in n or 'Conv' in n.split('+')[0]: c='conv'
        elif 'TopK' in n or 'topk' in n.lower(): c='topk'
        else: c='other'
        cats[c]=cats.get(c,0)+t
        mm=re.search(r'/model\.(\d+)/',n)
        if mm:
            k=int(mm.group(1))
            if k<=5: stem+=t
            elif k<=22: tail+=t
            else: head+=t
    print(path.split('/')[-1], 'total %.3f'%tot, {k:round(v,3) for k,v in cats.items()}, 'layers0-5 %.3f 6-22 %.3f head23 %.3f'%(stem,tail,head))
for p in sys.argv[1:]: split(p)
