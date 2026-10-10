# Per-layer cost of a runtime (gathered) kernel vs the static kernel, T4 profiles from receipts_phase1 (same exporter).
import re, sys, collections
P='/data/tmp/ds-yolo/seminar5/inputs/receipts_phase1/t4/'
def load(f):
    d={}
    for l in open(P+f):
        m=re.match(r'\s+([\d.]+)\s+(/model\.(\d+)/\S+/Conv)\b',l)
        if m: d[m.group(2)]=(float(m.group(1)),int(m.group(3)))
    return d
ref=load('profile_yolo26m_ref.txt')
for f in sys.argv[1:]:
    wb=load(f); lv=collections.defaultdict(lambda:[0,0,0])
    for k,(t,layer) in wb.items():
        if k in ref and 6<=layer<=22 and t>ref[k][0]*1.5:
            lvl={6:'P4',7:'P5',8:'P5',9:'P5',10:'P5',11:'up',12:'P4',13:'P4',14:'up',15:'P3',16:'P3',17:'P3',18:'P4',19:'P4',20:'P4',21:'P5',22:'P5'}[layer]
            lv[lvl][0]+=ref[k][0]; lv[lvl][1]+=t; lv[lvl][2]+=1
    print(f)
    for k,(a,b,n) in sorted(lv.items()):
        print(f'  {k}: {n} routed convs, static {a:.3f} ms -> runtime kernel {b:.3f} ms (x{b/a:.2f}), +{(b-a)/n:.4f} ms per layer')
