"""Phase 0 CPU correctness checks (no timing is reported from this box).
1 YAML build; 2 dense->bank conversion parity; 3 fuse parity; 4 deployment-graph parity; 5 deepcopy / pickle."""
import copy, glob, io, sys
import numpy as np, torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.moe.weight_bank import convert_to_weight_bank, bank_modules
from ultralytics.data.augment import LetterBox
import cv2

torch.manual_seed(0)
W = sys.argv[1] if len(sys.argv) > 1 else "/data/yolo-quant-work/weights/yolo26m.pt"
files = sorted(glob.glob("/data/datasets/coco/images/val2017/*.jpg"))[::500][:8]
lb = LetterBox((640, 640), auto=False)
ims = torch.stack([torch.from_numpy(lb(image=cv2.imread(f))[..., ::-1].transpose(2, 0, 1).copy()).float() / 255 for f in files])

def out(model, x):
    """Inputs of the Detect head (layers 16, 19, 22), flattened. The end2end output is a top-300 list whose row
    order flips on 1e-7 score differences, so it is not a usable parity metric."""
    feats = []
    h = model.model[-1].register_forward_pre_hook(lambda m, a: feats.append(torch.cat([t.flatten(1) for t in a[0]], 1).detach()))
    model(x); h.remove()
    return feats[0]

def rel(a, b):
    return float((a - b).abs().max() / b.abs().max())

# 1. YAML build
m_yaml = DetectionModel("yolo26m-wb.yaml", ch=3, nc=80, verbose=False)
banks = bank_modules(m_yaml)
n_dense = sum(p.numel() for p in DetectionModel("yolo26m.yaml", ch=3, nc=80, verbose=False).parameters())
n_bank = sum(p.numel() for p in m_yaml.parameters())
routed = sum(b.out_channels * b.in_channels for b in banks)
n_router = sum(p.numel() for b in banks if b.router is not None for p in b.router.parameters())
print(f"1. YAML build: {len(banks)} banked convs, routed params per expert {routed:,}, router params {n_router:,}")
print(f"   params dense {n_dense:,} -> bank {n_bank:,}; expected dense + 3 x routed + router = {n_dense + 3 * routed + n_router:,}  match={n_bank == n_dense + 3 * routed + n_router}")
print(f"   router owner: member 0 only = {[b.index for b in banks if b.router is not None] == [0]}; router input channels {banks[0].router.c1}")

# 2. conversion parity on real weights
dense = YOLO(W).model.float().eval()
ref = out(dense, ims)
wb = copy.deepcopy(dense)
state = convert_to_weight_bank(wb.model, 6, 22, experts=4, top_k=2, eps=0.01)
wb.yaml["weight_bank"] = dict(layers=[6, 22], experts=4, top_k=2, hidden=64, eps=0.0)
wb.eval()
y = out(wb, ims)
print(f"2. dense -> bank (eps 0.01, top-2, random router): max rel output diff {rel(y, ref):.3e}; logits std across 8 images {state.logits.std(0).mean():.4f}")
owner = bank_modules(wb)[0]
saved = [p.detach().clone() for p in owner.router.fc2.parameters()]
for p in owner.router.fc2.parameters():
    torch.nn.init.zeros_(p)
state.top_k = 4
print(f"   uniform gates (zero logits, k=E): max rel output diff {rel(out(wb, ims), ref):.3e}  (must be ~1e-6: exact reproduction)")
state.top_k = 2
for p, s in zip(owner.router.fc2.parameters(), saved):
    p.data.copy_(s)
y = out(wb, ims)

# 3. fuse parity
fused = copy.deepcopy(wb).fuse(verbose=False)
print(f"3. fuse: max rel diff fused vs unfused {rel(out(fused, ims), y):.3e}; is_fused={fused.is_fused()}")

# 4. deployment graph (batch 1) vs training/eval path, per k and lowering
for k in (2, 1, 4):
    fused.model[6].cv1.conv.state.top_k = k
    st = fused.model[6].cv1.conv.state
    for low in ("conv", "matmul"):
        d = []
        for i in range(4):
            st.lowering = None; a = out(fused, ims[i : i + 1])
            st.lowering = low; b = out(fused, ims[i : i + 1])
            d.append(rel(b, a))
        st.lowering = None
        print(f"4. deploy k={k} lowering={low}: max rel diff vs eval path over 4 images {max(d):.3e}")
st.top_k = 2

# 5. deepcopy and pickle keep the shared state
c = copy.deepcopy(wb); bs = bank_modules(c)
print(f"5. deepcopy: one shared state={len({id(b.state) for b in bs}) == 1}, distinct from source={bs[0].state is not state}, members={len(bs[0].state.members)}")
buf = io.BytesIO(); torch.save({"model": c}, buf); buf.seek(0)
r = torch.load(buf, weights_only=False)["model"].eval()
print(f"   pickle round trip: max rel diff {rel(out(r, ims), y):.3e}; state shared={len({id(b.state) for b in bank_modules(r)}) == 1}")
# state_dict load into a YAML-built model (the trainer's path)
missing = m_yaml.load_state_dict(wb.state_dict(), strict=True)
m_yaml.eval(); print(f"   strict state_dict load into yolo26m-wb.yaml model: ok; max rel diff {rel(out(m_yaml, ims), y):.3e}")
