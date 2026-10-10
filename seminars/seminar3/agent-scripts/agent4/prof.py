"""Agent 4: group TensorRT per-layer profile times by top-level YOLO layer."""
import re, sys, collections
R = "/data/tmp/ds-yolo/seminar3/inputs/receipts/t4/"
def load(f):
    rows = []
    for l in open(R + f):
        if l.startswith("#"): tot = l.strip(); continue
        t, n = l.strip().split(None, 1); rows.append((float(t), n))
    return tot, rows
def group(rows):
    g = collections.Counter()
    for t, n in rows:
        mm = re.search(r"/model\.(\d+)/", n)
        if "TopK" in n or "topk" in n.lower(): g["topk"] += t
        elif mm: g[int(mm.group(1))] += t
        else: g["other"] += t
    return g
for f in sys.argv[1:]:
    tot, rows = load(f); g = group(rows)
    s = lambda a, b: sum(v for k, v in g.items() if isinstance(k, int) and a <= k <= b)
    nz = sum(1 for t, n in rows if t > 0)
    print(f"{f}: {tot} | rows {len(rows)} nonzero {nz} | L0-5 {s(0,5):.3f} L6-22 {s(6,22):.3f} L23 {s(23,23):.3f} topk {g['topk']:.3f} other {g['other']:.3f} | L6-10 {s(6,10):.3f} L11-22 {s(11,22):.3f} | sum {sum(g.values()):.3f}")
    if "-v" in f: pass
