#!/usr/bin/env python3
"""Round 2: resolution oracle (M at 512/640/768, S at 768) and shared-stem M/L oracle with random-share nulls.
Same convention as oracle.py (per-image proxy chooses; every AP is the exact global pycocotools AP of the mixture).
"""
import os, pickle, sys
import numpy as np

sys.path.insert(0, "/data/tmp/ds-yolo/seminar4/work/agent4")
import oracle as O  # noqa: E402

D, W = O.D, O.W
NEW = {
    "m512": (f"{D}/dumpml_yolo26m_512_coco.json", 3.781),
    "m640": (f"{D}/dumpml_yolo26m_coco.json", 5.355),
    "m768": (f"{D}/dumpml_yolo26m_768_coco.json", 6.959),
    "s768": (f"{D}/dumpml_yolo26s_768_coco.json", 3.617),
}
evs, proxy, LAT = {}, {}, {}
MODE = sys.argv[1]
KEEP = {"C": ["m512","m640","m768"], "S": ["s768","m640","m768"], "ML": []}[MODE]
for k, (p, lat) in NEW.items():
    if k not in KEEP:
        continue
    ck = f"{W}/eval_{k}.pkl"
    if os.path.exists(ck):
        evs[k], proxy[k] = pickle.load(open(ck, "rb"))
    else:
        print("== eval", k, flush=True)
        evs[k] = O.run_eval(p)
        evs[k].cocoDt = None
        proxy[k] = O.per_image_proxy(evs[k])
        pickle.dump((evs[k], proxy[k]), open(ck, "wb"))
    LAT[k] = lat
    print(k, "global AP", round(evs[k].stats[0], 4), "AP_S/M/L", np.round(evs[k].stats[3:6], 4), flush=True)
# round-1 single-label m and l for the shared-stem m/l row (branch costs from the moderator insert)
for k in ("ml" if MODE == "ML" else ""):
    e, p = pickle.load(open(f"{W}/eval_{k}.pkl", "rb"))
    evs[k], proxy[k] = e, p
LAT["m"], LAT["l"] = 4.85, 6.06  # if_depth_ml per-branch medians (moderator insert section 3)
rng = np.random.default_rng(0)


def solve(keys, budget, name):
    P = np.stack([proxy[k] for k in keys], 1)
    L = np.array([LAT[k] for k in keys])

    def pick(lam):
        return (P - lam * L).argmax(1)

    if budget is None:
        ch = pick(0.0)
    else:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            lam = (lo + hi) / 2
            if L[pick(lam)].mean() > budget:
                lo = lam
            else:
                hi = lam
        ch = pick(hi)
    avg = L[ch].mean()
    shares = {keys[j]: round(float((ch == j).mean()), 3) for j in range(len(keys))}
    ap, aps, apm, apl = O.mixed_ap(evs, [keys[j] for j in ch])
    bar = 0.5261 + 0.0102 * (avg - 5.36)
    print(f"ORACLE {name}: avg={avg:.3f} AP={ap:.4f} S/M/L={aps:.3f}/{apm:.3f}/{apl:.3f} front={bar:.4f} "
          f"margin={ap - bar:+.4f} shares={shares}", flush=True)
    # random-share null: same shares, images permuted
    chn = ch[rng.permutation(len(ch))]
    apn = O.mixed_ap(evs, [keys[j] for j in chn])[0]
    print(f"NULL   {name}: avg={L[chn].mean():.3f} AP={apn:.4f} oracle-null={ap - apn:+.4f}", flush=True)
    return ch


if __name__ == "__main__" and MODE == "C":
    solve(["m512", "m640", "m768"], 5.36, "C: m 512/640/768 at 5.36")
    solve(["m512", "m640", "m768"], None, "C: m 512/640/768 unconstrained")
    solve(["m512", "m768"], 5.36, "C: m 512/768 at 5.36")
    solve(["m640", "m768"], 5.36, "C: m 640/768 (budget not binding)")
if MODE == "S":
    solve(["s768", "m640", "m768"], 5.36, "C': s768/m640/m768 at 5.36")
if MODE == "ML":
    ch = solve(["m", "l"], 5.36, "shared-stem m/l at 4.85/6.06, budget 5.36")
    # fixed-quantile 42% L share by a GT-free signal: m's own detection count (from the m640 dump proxies we cannot
    # read counts; use the per-image number of dets >0.25 from the evalImgs of m)
    ev = evs["m"]
    A = len(ev.params.areaRng)
    cnt = np.zeros(O.NI)
    for i in range(O.NI):
        for c in range(O.NC):
            e = ev.evalImgs[c * A * O.NI + i]
            if e is not None:
                cnt[i] += (np.array(e["dtScores"]) > 0.25).sum()
    order = np.argsort(-cnt)
    choice = np.array(["m"] * O.NI, dtype=object)
    choice[order[: int(0.42 * O.NI)]] = "l"
    ap = O.mixed_ap(evs, list(choice))[0]
    print(f"SIGNAL shared-stem m/l, top-42% by m's det count>0.25 -> L tail: AP={ap:.4f} (avg 5.36 ms)", flush=True)
    choice = np.array(["m"] * O.NI, dtype=object)
    choice[rng.permutation(O.NI)[: int(0.42 * O.NI)]] = "l"
    print(f"NULL   shared-stem m/l random 42% L: AP={O.mixed_ap(evs, list(choice))[0]:.4f}", flush=True)
