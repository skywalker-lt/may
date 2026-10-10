"""Round 4 check for vote V4: is every routed 1x1 bank followed by a BatchNorm2d in training mode?

If yes, the loss is invariant to a common rescaling of all kernels of a bank (base and deltas together), so uniform
weight decay leaves the delta-to-base ratio unchanged; decay on only one of the two does not.
"""
import torch
import torch.nn as nn

torch.set_num_threads(2)
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.moe.weight_bank import BankConv2d
from ultralytics.nn.tasks import DetectionModel

m = DetectionModel("/data/YOLO-Master/ultralytics/cfg/models/26/yolo26-wb.yaml", ch=3, nc=80, verbose=False)
banks = {n: mod for n, mod in m.named_modules() if isinstance(mod, BankConv2d)}
parents = dict(m.named_modules())
with_bn, without = [], []
for n in banks:
    parent = parents[n.rsplit(".", 1)[0]]
    ok = isinstance(parent, Conv) and n.endswith(".conv") and isinstance(getattr(parent, "bn", None), nn.BatchNorm2d)
    (with_bn if ok else without).append(n)
print("banks:", len(banks), "followed by BatchNorm2d inside a Conv:", len(with_bn), "not:", without)
print("banks with their own bias before fusing:", sum(b.bias is not None for b in banks.values()))

# Decay arithmetic over a schedule (per-step factor 1 - lr*wd), unsupported component, as in rounds 2-3:
for name, keep in (("80 ep upcycled", 0.986), ("600 ep scratch", 0.061)):
    print(f"{name}: equal decay   -> base x{keep}, delta x{keep}, ratio x1.000")
    print(f"{name}: delta-only    -> ratio x{keep:.3f}")
    print(f"{name}: delta no-decay, base decayed -> ratio x{1 / keep:.2f} (upper bound: base gradient support ignored)")
