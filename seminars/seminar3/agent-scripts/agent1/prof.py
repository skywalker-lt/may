import re, sys, collections
D = "/data/tmp/ds-yolo/seminar3/inputs/receipts/t4/"
def load(f):
    out = []
    for ln in open(D + f):
        m = re.match(r"\s*([0-9.]+)\s+(.*)", ln)
        if m: out.append((float(m.group(1)), m.group(2)))
    return out
def reg(name):
    ls = [int(x) for x in re.findall(r"/model\.(\d+)/", name)]
    if not ls: return "other(post/router/reformat)"
    l = min(ls)
    return "stem 0-5" if l <= 5 else "bb 6-10" if l <= 10 else "neck 11-16" if l <= 16 else "neck 17-22" if l <= 22 else "detect 23"
for f in sys.argv[1:]:
    rows = load(f); s = collections.Counter(); n = collections.Counter()
    for t, nm in rows:
        if t > 0: s[reg(nm)] += t; n[reg(nm)] += 1
    tot = sum(s.values())
    print(f, f"total {tot:.3f} ms, nonzero layers {sum(n.values())}, zero-time lines {sum(1 for t,_ in rows if t==0)}")
    for k in ("stem 0-5", "bb 6-10", "neck 11-16", "neck 17-22", "detect 23", "other(post/router/reformat)"):
        print(f"   {k:28s} {s[k]:.3f} ms  n={n[k]}")
    oth = sorted([(t, nm) for t, nm in rows if reg(nm).startswith("other") and t > 0.01], reverse=True)[:6]
    for t, nm in oth: print(f"      {t:.4f} {nm[:110]}")
