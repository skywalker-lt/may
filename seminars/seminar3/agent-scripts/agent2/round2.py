"""Agent 2, round 2: checks of the other agents' numbers (CPU, 2 threads)."""
import math, torch
torch.set_num_threads(2)
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.block import Attention
m = DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False).eval().fuse(verbose=False)
att = []
def hk(mod, i, o):
    B, C, H, W = i[0].shape; N = H * W
    att.append((C, N, mod.num_heads, mod.key_dim, mod.head_dim, mod.num_heads * N * N * (mod.key_dim + mod.head_dim)))
for mod in m.modules():
    if isinstance(mod, Attention): mod.register_forward_hook(hk)
with torch.no_grad(): m(torch.zeros(1, 3, 640, 640))
print("attention blocks (C,N,heads,key,head,MACs):", att, "sum G", sum(a[-1] for a in att) / 1e9, "-> 34.090 +", sum(a[-1] for a in att) / 1e9)
# rank-16 held: shared A (16*sum cin) + E * 16*sum cout
print("rank16 held M: A shared", 20.411132 + 0.282624 + 4 * 0.2048 + 0.033092, " A per expert (agent 3)", 20.411132 + 4 * 0.487424 + 0.033092)
H = lambda c: -sum(x / sum(c) * math.log2(x / sum(c)) for x in c)
for lab, c in (("fp32 val5000", (2323, 376, 1132, 1169)), ("fp16 val5000", (2294, 382, 1143, 1181)), ("agent4 n=250", (129, 9, 47, 65))):
    print(lab, "entropy bits %.3f" % H(c), "min share %.3f" % (min(c) / sum(c)), "min images/epoch %d" % (118287 * min(c) / sum(c)))
se = math.sqrt(0.0752 * (1 - 0.0752) / 250); print("agent4 9/250=0.036 vs 0.0752: z = %.2f" % ((0.036 - 0.0752) / se))
F = lambda L, L0=5.36: 0.5261 + 0.0102 * (L - L0)
print("bar literal A %.4f | bar vs dense-in-If twin (5.213 anchor) %.4f | dense-in-If under literal F: F=%.4f, public AP 0.5261 exceeds by %.4f" % (F(5.182) + .003, F(5.182, 5.213) + .003, F(5.213), 0.5261 - F(5.213)))
print("A6 held M", 20.411132 - 6.959616 + 6 * 6.959616 + 0.033092, "engine est MB", 2 * (20.411132 - 6.959616 + 6 * 6.959616 + 0.033092) + 4.2)
print("engine overhead MB: tied", 86.8 - 2 * 41.323, "full", 146.0 - 2 * 70.94, "dense4", 45.2 - 2 * 20.411)
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
def p(mu, sd, thr): return 1 - Phi((thr - mu) / sd)
for lab, ctl, sdc, d, sdd in (("(i) A-tied", 0.523, 0.0027, 0.0, 0.0020), ("(ii) A-tied", 0.516, 0.0045, 0.001, 0.0045), ("(ii) A6", 0.516, 0.0045, 0.0015, 0.0045), ("(i) dense-in-If control", 0.523, 0.0027, 0.0, 1e-9)):
    sd = math.hypot(sdc, sdd)
    print(lab, "P(literal bar .5273) %.3f  P(twin bar .5288) %.3f  P(delta>=.003) %.3f" % (p(ctl + d, sd, 0.5273 if 'dense' not in lab else F(5.213) + .003), p(ctl + d, sd, 0.5288), p(d, math.hypot(sdd, 0.0021), 0.003)))
