"""Round-2 check: is the gray-subset gap a content confound?  Colour subsets matched to the gray subset's
joint histogram of GT object count and small-object count (stratified draw), exact pycocotools AP for M@640
and L@640 (seminar-5 evalImgs caches, read-only).  One thread."""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7")
import numpy as np, mixlib
t0 = time.time()
ids = np.array(mixlib.img_ids()); gray = set(json.load(open("/data/tmp/ds-yolo/seminar6/work/agent6/gray_ids.json")))
g = np.array([i in gray for i in ids]); gi = np.where(g)[0]; ci = np.where(~g)[0]
n, ns, area = mixlib.gt_counts()
print(f"gray: mean count {n[gi].mean():.2f} small {ns[gi].mean():.2f} median-area {np.median(area[gi]):.0f} | "
      f"colour: {n[ci].mean():.2f} {ns[ci].mean():.2f} {np.median(area[ci]):.0f}", flush=True)
cb = np.digitize(n, [1, 3, 6, 11, 21]); sb = np.digitize(ns, [1, 3, 8])
strat = cb * 10 + sb
names = ["dumpml_yolo26m_coco", "dump_yolo26l_coco"]
def ap(model, sel): return mixlib.mix_ap_index(names, np.full(len(ids), model), sel)
mg, lg = ap(0, gi), ap(1, gi)
rng = np.random.default_rng(11); dm, dl = [], []
keys, cnts = np.unique(strat[gi], return_counts=True)
pools = {k: ci[strat[ci] == k] for k in keys}
for r in range(40):
    s = np.concatenate([rng.choice(pools[k], size=c, replace=len(pools[k]) < c) for k, c in zip(keys, cnts)])
    s = np.sort(s); dm.append(ap(0, s)); dl.append(ap(1, s))
dm, dl = np.array(dm), np.array(dl); d = dl - dm
print(f"count-matched colour subsets n={len(gi)} x40: M {dm.mean():.4f} sd {dm.std():.4f} | L {dl.mean():.4f} sd {dl.std():.4f} | L-M {d.mean():+.4f} sd {d.std():.4f}  [{time.time()-t0:.0f}s]")
print(f"gray minus matched: M {mg-dm.mean():+.4f} ({(mg-dm.mean())/dm.std():+.1f} sd), L {lg-dl.mean():+.4f} ({(lg-dl.mean())/dl.std():+.1f} sd); "
      f"L-M on gray {lg-mg:+.4f} vs matched {d.mean():+.4f}", flush=True)
