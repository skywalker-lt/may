"""Agent 3 (seminar 5) R0: initial checkpoints for branch B (no P3) and branch C (P2 leaf) from the public YOLO26-M.
usage: PYTHONPATH=/data/YOLO-Master python build_branches.py <yolo26m.pt> <c17.pt> <out dir>
  c17.pt = per-channel mean of public layer 17's output on 200 train2017 images (work/agent3/r2/levelvar.py calib 200).
Writes b_init.pt (yolo26m-p45.yaml), c_init.pt (yolo26m-p2leaf.yaml) and checks:
  B: b_init's P4/P5 outputs equal public layers 19/22 with layer 17 replaced by c17 (the fold is exact);
  C: c_init's layers 0-22 equal the public model bit for bit (the trunk and A's head are untouched)."""
import sys, copy
from pathlib import Path
import torch
from ultralytics.nn.tasks import DetectionModel
HERE = Path(__file__).resolve().parent
def load(pt):
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    return ck, (ck.get("ema") or ck["model"]).float().eval()
def save(ck, model, out):
    o = {k: v for k, v in ck.items() if k not in ("model", "ema", "optimizer", "scaler", "updates")}
    o.update(model=model, ema=None, optimizer=None, updates=None, epoch=-1, best_fitness=None)
    torch.save(o, out); print("saved", out)
if __name__ == "__main__":
    m_pt, c17_pt, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
    ck, M = load(m_pt); c17 = torch.load(c17_pt).float(); P = M.state_dict()
    # ---- B: public 0-13 -> 0-13; public 19..22 -> 14..17; Detect levels 1,2 -> 0,1
    B = DetectionModel(str(HERE / "yolo26m-p45.yaml"), ch=3, nc=80, verbose=False)
    Md = M.model[-1]
    for nm in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):   # keep the public head widths (Detect sizes c2/c3 by ch[0])
        setattr(B.model[18], nm, torch.nn.ModuleList([copy.deepcopy(getattr(Md, nm)[i]) for i in (1, 2)]))
    sd = {}; mapl = {19: 14, 20: 15, 21: 16, 22: 17}
    for k, v in P.items():
        i = int(k.split(".")[1]); rest = ".".join(k.split(".")[2:])
        if i <= 13: sd[k] = v.clone()
        elif i in mapl: sd[f"model.{mapl[i]}.{rest}"] = v.clone()
        elif i == 23:
            parts = rest.split(".")
            if parts[0] in ("cv2", "cv3", "one2one_cv2", "one2one_cv3") and parts[1] in ("1", "2"):
                parts[1] = str(int(parts[1]) - 1); sd["model.18." + ".".join(parts)] = v.clone()
            elif parts[0] not in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"): sd["model.18." + rest] = v.clone()
    # fold the constant half of public 19.cv1's input (channels 0..255 = layer 17) into its BN mean
    w = P["model.19.cv1.conv.weight"]                      # [c, 768, 1, 1]
    sd["model.14.cv1.conv.weight"] = w[:, 256:].clone()
    sd["model.14.cv1.bn.running_mean"] = P["model.19.cv1.bn.running_mean"] - w[:, :256, 0, 0] @ c17
    missing = B.load_state_dict(sd, strict=False); print("B load:", missing)
    B.names = M.names; B.args = getattr(M, "args", None); B.eval()
    # check against the public model with layer 17 replaced by the constant
    with torch.no_grad():
        xin = torch.rand(1, 3, 320, 320); yp, yb = [], []
        x = xin
        for m in M.model[:-1]:
            xi = x if m.f == -1 else (yp[m.f] if isinstance(m.f, int) else [x if j == -1 else yp[j] for j in m.f])
            x = c17.view(1, -1, 1, 1).expand(1, 256, 20, 20) if m.i == 17 else m(xi); yp.append(x)
        x = xin
        for m in B.model[:-1]:
            xi = x if m.f == -1 else (yb[m.f] if isinstance(m.f, int) else [x if j == -1 else yb[j] for j in m.f])
            x = m(xi); yb.append(x)
        print("B fold check: max |P4 diff| %.2e, max |P5 diff| %.2e" % ((yb[14] - yp[19]).abs().max(), (yb[17] - yp[22]).abs().max()))
    save(ck, B, out / "b_init.pt")
    # ---- C: public 0-22 unchanged; Detect: P3/P4/P5 heads from public levels 0-2 -> 1-3; P2 head fresh (bias init)
    Cm = DetectionModel(str(HERE / "yolo26m-p2leaf.yaml"), ch=3, nc=80, verbose=False)
    for nm in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):   # P2 head fresh (its own widths), P3-P5 heads public
        setattr(Cm.model[26], nm, torch.nn.ModuleList([getattr(Cm.model[26], nm)[0]] + [copy.deepcopy(getattr(Md, nm)[i]) for i in (0, 1, 2)]))
    sd = {}
    for k, v in P.items():
        i = int(k.split(".")[1]); rest = ".".join(k.split(".")[2:])
        if i <= 22: sd[k] = v.clone()
        elif i == 23:
            parts = rest.split(".")
            if parts[0] in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
                parts[1] = str(int(parts[1]) + 1); sd["model.26." + ".".join(parts)] = v.clone()
            else: sd["model.26." + rest] = v.clone()
    missing = Cm.load_state_dict(sd, strict=False)
    print("C load: missing", len(missing.missing_keys), "(P2 leaf + P2 head, fresh), unexpected", len(missing.unexpected_keys))
    Cm.names = M.names; Cm.args = getattr(M, "args", None); Cm.eval()
    with torch.no_grad():
        xin = torch.rand(1, 3, 320, 320)
        ya, yc, x1, x2 = [], [], xin, xin
        for m in M.model[:-1]:
            xi = x1 if m.f == -1 else (ya[m.f] if isinstance(m.f, int) else [x1 if j == -1 else ya[j] for j in m.f]); x1 = m(xi); ya.append(x1)
        for m in Cm.model[:23]:
            xi = x2 if m.f == -1 else (yc[m.f] if isinstance(m.f, int) else [x2 if j == -1 else yc[j] for j in m.f]); x2 = m(xi); yc.append(x2)
        print("C trunk check: layers 0-22 identical:", all(torch.equal(p, q) for p, q in zip(ya, yc)))
    save(ck, Cm, out / "c_init.pt")
