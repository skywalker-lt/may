"""Agent 2: extra MACs of the residual variants from count.log's 39-layer table; latency bars F(L)+0.003."""
import re
rows = [l.split() for l in open("count.log") if l.startswith("   model.")]
L = [(r[0], int(r[1]), int(r[3]), int(r[5])) for r in rows]  # name, cin, cout, hw
out = [r for r in L if re.fullmatch(r"model\.\d+\.cv2\.conv", r[0])]
print("block-output convs", len(out), "channels", sum(r[2] for r in out), "out elems", sum(r[2] * r[3] ** 2 for r in out))
print("39 layers: channels", sum(r[2] for r in L), "out elems", sum(r[2] * r[3] ** 2 for r in L))
print("rank-16 extra MACs, 39 layers: %.4f G" % (sum(r[3] ** 2 * 16 * (r[1] + r[2]) for r in L) / 1e9),
      "| 8 outputs: %.4f G" % (sum(r[3] ** 2 * 16 * (r[1] + r[2]) for r in out) / 1e9),
      "| shared-A params", sum(16 * r[1] for r in L), "routed-B params/expert", sum(16 * r[2] for r in L))
for n, ms in [("A cond top-1", 5.181), ("B scale+shift x8", 5.721), ("B rank-16 x8", 5.861), ("C shift", 5.922), ("C scale+shift", 6.044),
              ("C rank-16", 6.759), ("D top-1 matmul", 5.953), ("D top-2 matmul", 6.130), ("D soft matmul", 6.258)]:
    F = 0.5261 + 0.0102 * (ms - 5.36)
    print(f"{n:18} L={ms:.3f} ms  ratio {ms/5.366:.3f}  F(L)={F:.4f}  bar={F+0.003:.4f}")
