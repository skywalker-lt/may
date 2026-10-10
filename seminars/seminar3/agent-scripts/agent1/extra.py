import json
rows = json.load(open("rows.json")); lay = lambda r: int(r["name"].split(".")[1])
one = [r for r in rows if r["kind"] == "conv" and r["k"] == 1 and r["g"] == 1 and 6 <= lay(r) <= 22]
for nm, lo, hi in (("6-10", 6, 10), ("11-16", 11, 16), ("17-22", 17, 22), ("11-22", 11, 22)):
    s = [r for r in one if lo <= lay(r) <= hi]
    print(nm, "1x1 n", len(s), "params M", sum(r["params"] for r in s) / 1e6, "MACs G", sum(r["macs"] for r in s) / 1e9)
oe = sum(r["cout"] * r["hw"] ** 2 for r in one); print("output elements of 39 convs M", oe / 1e6)
print("rank16 extra MACs G", sum(16 * r["cin"] * r["hw"] ** 2 + 16 * r["cout"] * r["hw"] ** 2 for r in one) / 1e9, "shared A params", sum(16 * r["cin"] for r in one))
out8 = [r for r in one if r["name"].count(".") == 3 and ".cv2." in r["name"]]
print("8 block outputs: n", len(out8), "elements M", sum(r["cout"] * r["hw"] ** 2 for r in out8) / 1e6)
tot_p = 20.411132; p1 = sum(r["params"] for r in one) / 1e6; br = 16.831
for nm, held in (("A 1x1-distinct E=4", tot_p + 3 * p1), ("A full branches E=4", 3.580 + 4 * br), ("A6 six merged-pair branches (1x1 distinct)", tot_p + 5 * p1), ("A E=8", tot_p + 7 * p1)):
    print(f"{nm}: held {held:.2f} M -> fp16 engine est {held*2:.0f} MB")
