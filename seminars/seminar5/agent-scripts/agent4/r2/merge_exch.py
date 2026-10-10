"""(1) Agent 7's exchange vs the plain routed shrink on ONE accounting (T4 unset basis of agent 7: M 5.36, L 6.89, M@512 3.78,
router 0.18 on every image). (2) Concentration of the attention proxy gain (YOLOv12-M-sdpa minus YOLO11-M, single-label) on
the images the shrink router sends to 512. Exact mixtures (agent 1's mixlib, imported read-only). CPU, one thread."""
import os, sys; os.environ['OMP_NUM_THREADS']='1'
sys.path.insert(0,'/data/tmp/ds-yolo/seminar5/work/agent1')
import numpy as np, mixlib as M, feats as F
ids, ns, nm, nl, minA = F.gt_counts(); n320, m640 = F.features(ids)
tot = ns+nm+nl
r_sm = F.oof_ridge(n320, np.log1p(ns+nm))        # agent 1's shrink router target
r_ct = F.oof_ridge(n320, np.log1p(tot))          # agent 7's count router target
rng = np.random.default_rng(0)
m512, m640e, l640 = M.evaluated('dumpml_yolo26m_512_coco'), M.evaluated('dumpml_yolo26m_coco'), M.evaluated('dump_yolo26l_coco')
assert list(m512['imgIds']) == [int(i) for i in ids]
front = lambda t: 0.5261 + 0.0102*(t-5.36)
def lowest(score, q):
    o = np.argsort(score + 1e-9*rng.random(len(score))); s = np.zeros(len(score), bool); s[o[:int(round(q*len(score)))]] = True; return s
def highest(score, q): return lowest(-score, q)
print('--- (1) exchange vs shrink, agent-7 accounting')
for q in (0.33, 0.40, 0.50):
    for name, r in (('count', r_ct), ('S+M', r_sm)):
        sh = lowest(r, q); ch = np.where(sh, 0, 1)
        t_sh = q*3.78 + (1-q)*5.36 + 0.18; ap_sh = M.score([m512, m640e], ch)[0]
        qL = (q*1.58 - 0.18)/1.53; Lm = highest(r, qL) & ~sh; ch2 = np.where(sh, 0, np.where(Lm, 2, 1))
        t_ex = q*3.78 + Lm.mean()*6.89 + (1-q-Lm.mean())*5.36 + 0.18; ap_ex = M.score([m512, m640e, l640], ch2)[0]
        nullL = np.zeros(len(r), bool); cand = np.where(~sh)[0]; nullL[rng.choice(cand, Lm.sum(), replace=False)] = True
        ap_exn = M.score([m512, m640e, l640], np.where(sh, 0, np.where(nullL, 2, 1)))[0]
        print(f'q={q:.2f} router={name:5s} shrink-only {ap_sh:.4f} @ {t_sh:.2f} ms (front+{ap_sh-front(t_sh):+.4f}) | exchange {ap_ex:.4f} @ {t_ex:.2f} ms (front{ap_ex-front(t_ex):+.4f}), L share {Lm.mean():.3f}; same shrink + L to RANDOM non-shrunk images {ap_exn:.4f} (L leg vs its own null {ap_ex-ap_exn:+.4f})', flush=True)
print('--- (2) attention proxy concentration on the shrink-routed images')
a11, a12 = M.evaluated('dump_yolo11m_coco'), M.evaluated('dump_yolov12m_sdpa_coco')
A11 = M.score([a11], np.zeros(len(ids)))[0]; A12 = M.score([a12], np.zeros(len(ids)))[0]; print('11m', A11, '12m', A12)
for q in (0.4, 0.5, 0.6):
    sh = lowest(r_sm, q); ap = M.score([a11, a12], sh.astype(int))[0]
    nl_ = [M.score([a11, a12], lowest(rng.random(len(ids)), q).astype(int))[0] for _ in range(3)]
    print(f'shrink share {q}: attention on shrunk images {ap:.4f}  phi={(ap-A11)/(A12-A11):.2f} | null {np.mean(nl_):.4f} phi={(np.mean(nl_)-A11)/(A12-A11):.2f} | lift {ap-np.mean(nl_):+.4f}', flush=True)
    ap_o = M.score([a11, a12], (~sh).astype(int))[0]; print(f'   attention on the NON-shrunk images instead: {ap_o:.4f} phi={(ap_o-A11)/(A12-A11):.2f}', flush=True)
