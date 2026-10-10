# Split agent 5's H200 TCR stub profiles into trunk / router+topk / dispatch / expert / combine (CPU, text only)
import re, sys
def cat(n):
    if re.search(r'/layers\.\d+/|one2one|Reformatting CopyNode', n) and not n.startswith('__my') : return 'trunk'
    if 'Topk' in n: return 'topk'
    if re.match(r'(Gemm_|_gemm_mha|__mye|__myl_FcAdd)', n) or re.search(r'MaxrSubExpSum|DivMulTran|Silu|AddSqrtDiv|MeanSub', n): return 'expert'
    if re.search(r'/MatMul|Gath|MoveResh|__myl_Tran_|MoveConc|ReshReshReshConc|ReplReshConc', n): return 'dispatch'
    return 'other'
for f in sys.argv[1:]:
    rows=[]
    for l in open(f):
        if l.startswith('#') or not l.strip(): continue
        t, n = l.strip().split(None,1); rows.append((float(t), n))
    tot=sum(t for t,_ in rows); agg={}; cnt={}
    for t,n in rows:
        c=cat(n); agg[c]=agg.get(c,0)+t; cnt[c]=cnt.get(c,0)+1
    print(f"== {f.split('/')[-1]} profiled total {tot:.3f} ms, {len(rows)} layers")
    for c in sorted(agg, key=lambda c:-agg[c]): print(f"  {c:9s} {agg[c]:.3f} ms ({agg[c]/tot*100:4.1f}%)  layers {cnt[c]}")
    print("  'other' layers:", [n[:60] for t,n in rows if cat(n)=='other' and t>0.004][:12])
