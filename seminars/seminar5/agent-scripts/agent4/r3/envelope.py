# Dense envelope over (model, input scale) at SA-2's operating point. Latencies est. by pixel scaling a+b(s/640)^2
# fitted on M's two measured T4 points; L assumed to share M's curve shape (scaled by its 640 time). AP measured (dumps).
import numpy as np
AP={('m',448):0.4937,('m',512):0.5063,('m',576):0.5203,('m',608):0.5234,('m',640):0.5261,
    ('l',448):0.5117,('l',512):0.5262,('l',576):0.5368,('l',640):0.5417,('s',640):0.4795,('n',640):0.4060}
def run(name,m512,m640,l640,s640,n640,rs2_ms,router):
    b=(m640-m512)/0.36; a=m640-b
    T={}
    for (f,s),ap in AP.items():
        base=a+b*(s/640)**2
        T[(f,s)]={'m':base,'l':base*l640/m640,'s':s640,'n':n640}[f] if f in 'ml' else {'s':s640,'n':n640}[f]
    pts=sorted((T[k],AP[k],k) for k in AP)
    hull=[]
    for p in pts:
        while len(hull)>=2 and (hull[-1][1]-hull[-2][1])*(p[0]-hull[-2][0])<=(p[1]-hull[-2][1])*(hull[-1][0]-hull[-2][0]): hull.pop()
        hull.append(p)
    env=lambda t: np.interp(t,[h[0] for h in hull],[h[1] for h in hull])
    F=lambda t: 0.5261+0.0102*(t-m640)
    print(f'== {name}: hull vertices', [(k,round(t,2),ap) for t,ap,k in hull])
    for lab,ms,ap in rs2_ms:
        e=env(ms); sl=(env(ms+0.05)-env(ms-0.05))/0.1
        print(f'{lab:44s} {ms:.2f} ms AP {ap:.4f} | bar F+0.003 {F(ms)+0.003:.4f} ({ap-F(ms)-0.003:+.4f}) | envelope {e:.4f} ({ap-e:+.4f}); env+0.003 needs {e+0.003:.4f}; env slope {sl:.4f}/ms')
# SA-2 = RS-2 at share 0.508 (0.5228 measured) + block on routed images (+0.11-0.22 ms T4 est.)
s=0.508
for basis,m512,m640,l640,lo_branch,hi_branch in (('unset, router on every image',3.78,5.36,6.89,3.78+0.18,5.36+0.18),
                                                ('real, router on every image (agent 1 mid)',3.29,5.05,6.55,3.47,5.05+0.178)):
    rs=s*lo_branch+(1-s)*hi_branch
    rows=[('RS-2 share 0.508 (measured)',rs,0.5228)]
    for blk in (0.11,0.22):
        for g in (0.000,0.002,0.006,0.010):
            rows.append((f'SA-2 block {blk} ms, dense gain g={g}',rs+s*blk,0.5228+0.60*g))
    run(basis,m512,m640,l640,2.75,1.65,rows,0.18)
print('\n#### RS-2L (public L at 512/640, same train-fitted route, share 0.508): measured 0.5390; nulls 0.5342 / 0.5318')
for basis,m512,m640,l640 in (('unset',3.78,5.36,6.89),('real',3.29,5.05,6.55)):
    k=l640/m640; b=(m640-m512)/0.36; a=m640-b; l512=(a+b*0.64)*k
    for lab,r in (('router on every image',0.18),('router free (in-If credit)',0.0)):
        ms=s*(l512+r)+(1-s)*(l640+r)
        run(f'{basis} {lab}',m512,m640,l640,2.75,1.65,[('RS-2L share 0.508',ms,0.5390),('RS-2L null mean',ms,0.5330)],r)
