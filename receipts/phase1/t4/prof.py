import sys, collections, re
def cat(n):
    if n.startswith("Reformatting"): return "reformat copies"
    if "Topk" in n or "TopK" in n: return "TopK"
    if n.startswith("__myl"):
        if "Gath" in n and ("Slic" in n or "Mul" in n) and len(n) > 60: return "router+gather+mix"
        if n.startswith("__myl_Slic") or n.startswith("__myl_Resh") or "SlicResh" in n: return "kernel slice/reshape/cast"
        if n.startswith("__myl_Cast"): return "cast"
        return "other myl"
    if "MatMul" in n or "matmul" in n.lower(): return "MatMul (routed 1x1)"
    if "Conv" in n and "PWN" in n: return "Conv+PWN fused"
    if "Conv" in n: return "Conv, no fused act"
    if n.startswith("PWN"): return "standalone PWN"
    return "other"
tab = collections.OrderedDict()
for p in sys.argv[1:]:
    c = collections.defaultdict(lambda: [0, 0.0]); tot = 0
    for l in open(p):
        if l.startswith("#"): continue
        t, name = l.strip().split(None, 1); t = float(t); k = cat(name); c[k][0] += 1; c[k][1] += t; tot += t
    c["TOTAL (profiling mode)"] = [sum(v[0] for v in c.values()), tot]; tab[p.replace("profile_", "").replace(".txt", "")] = c
keys = sorted({k for c in tab.values() for k in c}, key=lambda k: -max(c[k][1] if k in c else 0 for c in tab.values()))
print(f"{'category':28}" + "".join(f"{n[:16]:>18}" for n in tab))
for k in keys:
    print(f"{k:28}" + "".join(f"{(str(c[k][0]) + ' / ' + format(c[k][1], '.3f')) if k in c else '-':>18}" for c in tab.values()))
