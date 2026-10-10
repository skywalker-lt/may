# n320 features + label-derived counts on a random train2017 subset (for the train-fitted router, gate K2)
import os, sys, time, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__)); from n320feat import feat
N = int(sys.argv[1]); rng = np.random.RandomState(0)
imgs = sorted(os.listdir("/data/datasets/coco/images/train2017")); sel = rng.choice(len(imgs), N, replace=False)
F, A, ids = [], [], []; t0 = time.time()
for k, j in enumerate(sel):
    fn = imgs[j]; p = f"/data/datasets/coco/images/train2017/{fn}"; im = cv2.imread(p); h, w = im.shape[:2]
    F.append(feat(p)); lab = f"/data/datasets/coco/labels/train2017/{fn[:-4]}.txt"; areas = []
    if os.path.exists(lab):
        for line in open(lab):
            v = line.split()
            if len(v) >= 5: areas.append(float(v[3]) * w * float(v[4]) * h)
    A.append(np.array(areas, np.float32)); ids.append(int(fn[:-4]))
    if k % 2000 == 0: print(k, round(time.time() - t0), "s", flush=True)
np.savez("/data/tmp/ds-yolo/seminar5/work/agent1/r2/train_n320.npz", ids=np.array(ids), n320=np.stack(F), areas=np.array(A, dtype=object), allow_pickle=True)
print("done", round(time.time() - t0), "s")
