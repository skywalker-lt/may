"""Agent 2: split a TensorRT per-layer profile into stem (model.0-5), conditional branches, TopK, and the rest."""
import re, sys, collections
for p in sys.argv[1:]:
    stem = rest = topk = 0.0; br = collections.defaultdict(float)
    for l in open(p):
        if l.startswith("#"): continue
        t, n = l.strip().split(None, 1); t = float(t); m = re.search(r"/model\.(\d+)/", n)
        if "IfConditional" in n: br[re.search(r"\^e(\d)_", n).group(1)] += t
        elif "opk" in n: topk += t
        elif m and int(m.group(1)) <= 5: stem += t
        else: rest += t
    print(p.split("/")[-1], "stem %.3f topk %.3f rest %.3f branches" % (stem, topk, rest), {k: round(v, 3) for k, v in br.items()})
