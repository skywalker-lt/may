# Amdahl estimate of an INT8 YOLO26-M on the T4 / L4 from the per-layer fp16 profiles (est., no INT8 receipt exists).
import re,sys
def parts(path):
    segs={'stem':[0,0],'tail':[0,0],'head':[0,0],'none':[0,0]}  # [conv, nonconv]
    tot=0
    for line in open(path):
        m=re.match(r'\s*([\d.]+)\s+(.*)',line)
        if not m or line.startswith('#'): continue
        t=float(m.group(1)); n=m.group(2); tot+=t
        mm=re.search(r'/model\.(\d+)/',n); k=int(mm.group(1)) if mm else -1
        seg='none' if k<0 else ('stem' if k<=5 else ('tail' if k<=22 else 'head'))
        isconv=('Reformatting' not in n) and ('/conv/Conv' in n.split(' + ')[0] or n.split(' + ')[0].endswith('/Conv'))
        segs[seg][0 if isconv else 1]+=t
    return tot,segs
for path,label,real in [(sys.argv[1],'T4 yolo26m',5.36),(sys.argv[2],'L4 yolo26m',2.551),(sys.argv[3],'L4 yolo26l',3.346)]:
    tot,s=parts(path)
    print(label,'profile total %.3f'%tot,{k:(round(v[0],3),round(v[1],3)) for k,v in s.items()})
    conv=sum(v[0] for v in s.values()); non=tot-conv
    for cs,extra in [(0.50,0.03),(0.62,0.06)]:   # conv time ratio INT8/fp16, added Q/DQ reformat share of total
        t8=non+conv*cs+extra*tot
        print('   all-INT8: conv ratio %.2f, +%.0f%% reformat -> %.3f of fp16 = %.2f ms'%(cs,extra*100,t8/tot,t8/tot*real))
        # INT8 stem (layers 0-5 + unlabelled input ops), fp16 tail+head
        st=s['stem'][0]*cs+s['stem'][1]+s['none'][1]+s['none'][0]*cs
        tb=st+s['tail'][0]+s['tail'][1]+s['head'][0]+s['head'][1]+extra*tot*0.5
        print('   INT8 stem + fp16 tail: %.3f of fp16 = %.2f ms'%(tb/tot,tb/tot*real))
