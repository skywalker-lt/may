"""Paired check of the gray-domain reading in gray_domain.log: same random colour subsets for M and L (size 136),
and a bootstrap of L-M inside the gray subset. Uses seminar-5 agent 7's evalImgs caches (read-only) via mixlib;
exact pycocotools AP of image subsets. One thread."""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, "/data/tmp/ds-yolo/seminar5/work/agent7")
import numpy as np, mixlib
t0 = time.time()
ids = np.array(mixlib.img_ids()); gray = set(json.load(open("/data/tmp/ds-yolo/seminar6/work/agent6/gray_ids.json")))
g = np.array([i in gray for i in ids]); gi = np.where(g)[0]; ci = np.where(~g)[0]
names = ["dumpml_yolo26m_coco", "dump_yolo26l_coco"]
def ap(model, sel): return mixlib.mix_ap_index(names, np.full(len(ids), model), sel)
mg, lg = ap(0, gi), ap(1, gi)
print(f"gray n={len(gi)}: M {mg:.4f} L {lg:.4f} L-M {lg-mg:+.4f}  [{time.time()-t0:.0f}s]", flush=True)
rng = np.random.default_rng(7); dm, dl = [], []
for k in range(40):
    s = np.sort(rng.choice(ci, size=len(gi), replace=False)); dm.append(ap(0, s)); dl.append(ap(1, s))
dm, dl = np.array(dm), np.array(dl); d = dl - dm
print(f"colour subsets n={len(gi)} x40 (paired): M {dm.mean():.4f} sd {dm.std():.4f} | L {dl.mean():.4f} sd {dl.std():.4f} | "
      f"L-M {d.mean():+.4f} sd {d.std():.4f}  [{time.time()-t0:.0f}s]", flush=True)
print(f"gray minus matched colour: M {mg-dm.mean():+.4f} ({(mg-dm.mean())/dm.std():+.1f} sd), L {lg-dl.mean():+.4f} "
      f"({(lg-dl.mean())/dl.std():+.1f} sd); L-M gain gray minus colour {lg-mg-d.mean():+.4f} ({(lg-mg-d.mean())/d.std():+.1f} sd)", flush=True)
bd = []
for k in range(40):
    s = np.sort(rng.choice(gi, size=len(gi), replace=True)); bd.append(ap(1, s) - ap(0, s))
print(f"gray bootstrap x40 of L-M: mean {np.mean(bd):+.4f} sd {np.std(bd):.4f}  [{time.time()-t0:.0f}s]", flush=True)
