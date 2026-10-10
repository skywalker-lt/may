#!/usr/bin/env python3
"""Realistic router receipt (candidates A/F): can a thumbnail feature predict which model to run per image?
Feature: YOLO26-n's pooled layer-9 embedding (ultralytics `embed=[9]`) of a 160x160 thumbnail, CPU, 2 threads.
Router: ridge regression from the feature to the 5 per-image proxy APs, 5-fold CV over val2017 (train on 4000,
predict on 1000, never on its own images). Policy: Lagrangian pick on the PREDICTED proxies, lambda bisected so the
average T4 latency is 5.36 ms, allowed sets n/m/l and n/s/m/l/x. Reported AP is the exact mixed pycocotools AP."""
import os, sys, pickle, numpy as np, torch
torch.set_num_threads(2)
import oracle as O
from ultralytics import YOLO

W = O.W
feat_path = f"{W}/feat_n160.npy"
if not os.path.exists(feat_path):
    model = YOLO("/data/yolo-quant-work/weights/yolo26n.pt")
    imgdir = "/data/datasets/coco/images/val2017"
    feats = []
    B = 50
    for b in range(0, O.NI, B):
        paths = [f"{imgdir}/{iid:012d}.jpg" for iid in O.imgIds[b:b + B]]
        out = model.predict(paths, embed=[9], imgsz=160, device="cpu", verbose=False)
        feats.append(torch.stack(out).float().numpy() if isinstance(out, list) else out.float().numpy())
        if b % 1000 == 0:
            print("feat", b, flush=True)
    np.save(feat_path, np.concatenate(feats))
X = np.load(feat_path)
X = (X - X.mean(0)) / (X.std(0) + 1e-6)
evs, proxy = pickle.load(open(f"{W}/evals.pkl", "rb"))
keys = list(O.MODELS)
P = np.load(f"{W}/proxy.npy")
L = np.array([O.LAT[k] for k in keys])
rng = np.random.default_rng(0)
folds = np.array_split(rng.permutation(O.NI), 5)
Phat = np.zeros_like(P)
for f in folds:
    tr = np.setdiff1d(np.arange(O.NI), f)
    A = np.c_[X[tr], np.ones(len(tr))]
    lam = 300.0
    Wt = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ P[tr])
    Phat[f] = np.c_[X[f], np.ones(len(f))] @ Wt
gain_true, gain_hat = P[:, 3] - P[:, 0], Phat[:, 3] - Phat[:, 0]
print("CV corr(pred gain l-n, true gain l-n) =", np.corrcoef(gain_true, gain_hat)[0, 1])
print("CV corr per model:", [round(float(np.corrcoef(P[:, j], Phat[:, j])[0, 1]), 3) for j in range(5)])


def pick(S, lam, allowed):
    sc = S - lam * L
    sc[:, [j for j in range(5) if keys[j] not in allowed]] = -1e9
    return sc.argmax(1)


for allowed in ["nml", "nsmlx", "ml", "nl"]:
    lo, hi = 0.0, 1.0
    for _ in range(60):
        lam = (lo + hi) / 2
        if L[pick(Phat, lam, allowed)].mean() > 5.36:
            lo = lam
        else:
            hi = lam
    ch = pick(Phat, hi, allowed)
    shares = {keys[j]: round(float((ch == j).mean()), 3) for j in range(5) if (ch == j).any()}
    ap, aps, apm, apl = O.mixed_ap(evs, [keys[j] for j in ch])
    print(f"ROUTER_CV allowed={allowed} avg_lat={L[ch].mean():.3f} AP={ap:.4f} S/M/L={aps:.3f}/{apm:.3f}/{apl:.3f} margin_vs_M={ap-0.5261:+.4f} shares={shares}", flush=True)
