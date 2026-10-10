import re, pathlib
p = pathlib.Path("/data/tmp/ds-yolo/seminar3/round4/agent3.md")
t = p.read_text()
L = t.split("\n")
i0 = next(i for i, l in enumerate(L) if l.startswith("| failure mode"))
i1 = next(i for i, l in enumerate(L) if l.startswith("| weight decay on deltas"))
new = """| failure mode | `dense control` | `A-tied` | `A6` | `A-full` | `B` and `C` | `D` |
|---|---|---|---|---|---|---|
| fp16 route flips | none / low | 36 of 5,000, margins to 0.0090 / low / AP effect under 0.0005 est.; a 0.03 margin closes it | 54 set flips / low / half the kernels change | 36 / low | unm. / low | unm. / low |
| mirror and near-tie instability | none / low | 20.1% change branch / medium / needs label inheritance, not a margin | 24.6% change pair, 12.4% of kernels / medium | 20.1% / high / 82% of read weights switch | unm. / low | 24.6% (pair) / low |
| unfamiliar images | unm. / low | unm. / medium / wrong expert | unm. / medium | unm. / high | unm. / low | unm. / low |
| expert imbalance or collapse | none / low | 7.5% smallest share at step 0 / medium / collapse returns dense | 25.8% smallest kernel share, 4.3% smallest pair / medium | 7.5% / high / 16.8M values on a quarter of the data | unm. / low | released routers ended constant / medium |
| train/deploy mismatch | none / low | unm. / low / trained as deployed | ORT matches PyTorch to 6e-5 / low | unm. / low | unm. / low | unm. / high / soft top-2 to hard |
| quantisation (INT8) | unm. / low | unm. / medium / smallest branch gets 7.5% of calibration images | unm. / medium / smallest pair 4.3% | unm. / medium | unm. / low | unm. / medium |
| batch above 1 | none / low | not measurable, scalar `If` / high for batched serving, none on the device of record | same | same | unm. / low | unm. / low |
| engine size, build variance | 41 MB; `If` rebuilds 5.204 / 5.199 ms / low | 86.8 MB; rebuilds 5.182 / 5.178 ms / low | 116.6 MB / low | 146.0 MB / medium | 41-44 MB / low | 83 MiB / low |
| weight decay on deltas | none / low | calculated (`dq`), unm. / low with equal decay; delta-only decay removes 50-99.6% of an unsupported delta in 600 epochs | same | unm. / low | unm. / high at 600 epochs / routed terms erode | unm. / low / router fix mandatory |""".split("\n")
L[i0:i1 + 1] = new
t = "\n".join(L)
R = [
("Sources: `R` = the Phase 0-1 receipts; `M2`, `M3` = the moderator inserts of rounds 2 and 3; `r3`, `dq` =\n`work/agent3/round3.log`, `decay_equiv.log`. Latency is T4 only. Every AP of a routed model is a prediction.",
 "Sources: `R` = the Phase 0-1 receipts; `M2`, `M3` = the moderator inserts; `r3`, `dq` = `work/agent3/round3.log`,\n`decay_equiv.log`. Latency is T4 only. Every AP of a routed model is a prediction."),
("- **V1: (c)** A-tied and A6 both: same measured latency (5.181 and 5.175 ms, `M3`), and they differ in the one thing\n  a pilot can test, images per kernel set (29.6k against 59.1k per epoch).",
 "- **V1: (c)** both: same measured latency (5.181 and 5.175 ms, `M3`), differing in the one thing a pilot can test,\n  images per kernel set (29.6k against 59.1k per epoch)."),
("- **V2: (c)** fixed targets first: they start at equal shares and need no gate gradient, and a straight-through gate\n  is unlikely to find a useful partition where a fixed one shows none.",
 "- **V2: (c)** fixed targets start at equal shares and need no gate gradient, and a straight-through gate is\n  unlikely to find a useful partition where a fixed one shows none."),
("the\n  rule must still be written before the run, in units of the measured spread.",
 "the\n  rule is written before the run, in units of the measured spread."),
("Twin: dense-in-`If` (5.206 ms in the same rounds) for the A rows; same-exporter YOLO26-M (5.507 ms, `R`) for B, C,\nD. The four files agree on every count. No row executes fewer MACs than YOLO26-M.",
 "Twin: dense-in-`If` (5.206 ms, same rounds) for the A rows; same-exporter YOLO26-M (5.507 ms, `R`) for B, C, D.\nNo row executes fewer MACs than YOLO26-M."),
("Cells: measured value / severity / reason; \"unm.\" = unmeasured. Severities assume the rule of section 3; as coded\ntoday, collapse and train/deploy mismatch are high for every A row (zero router gradient at top-1).",
 "Cells: measured value / severity / reason; \"unm.\" = unmeasured. Severities assume the rule of section 3; as coded\ntoday, collapse and train/deploy mismatch are high for the A rows (zero router gradient at top-1)."),
("For\n`dense control` the third column is the chance that the wrapped dense engine alone reaches 0.5288. Round-3\nconsensus where the four files agree, my values otherwise.",
 "For\n`dense control` the third column is the wrapped dense engine against 0.5288. Round-3 consensus where the four\nfiles agree, my values otherwise."),
(" Every report\ncarries shares, mirror agreement, the delta-to-base norm ratio, and the permuted-route, merged-constant and\nforced-single-branch scores.",
 " Every report\ncarries shares, mirror agreement, the delta-to-base norm ratio, and the permuted-route and merged-constant scores."),
("- **Headroom stop rule (V5).** \"No fixed threshold\" must not mean a judgement made after the numbers arrive. I sign\n  it only with the rule of step 1 fixed beforehand in units of s.\n", ""),
("Scored routed, with\n   permuted routes, as the six pair averages, and with a lowest-loss oracle per partition.",
 "Scored routed, with\n   permuted routes, as six pair averages, and with a lowest-loss oracle."),
(" A positive gap under 0.003 with clean mechanics earns one extension to 20 epochs;\n   anything else stops at E.",
 " A positive gap under 0.003 with clean mechanics earns one extension to 20 epochs;\n   otherwise E."),
(" Otherwise stop at E\n   and report the pair as the negative.", " Otherwise E, with the pair\n   reported as the negative."),
("Step 0 is useful under every outcome,\nand step 1 costs a few fine-tunes of 7M parameters to learn whether anything is there. I would run those two,\nexpect to stop, and treat a pass at step 1 as a surprise that has earned the next gate.",
 "Step 0 is useful under every outcome,\nand step 1 costs a few fine-tunes of 7M parameters. I would run those two, expect to stop, and treat a pass at\nstep 1 as a surprise that has earned the next gate."),
]
for a, b in R:
    assert a in t, a[:60]
    t = t.replace(a, b)
p.write_text(t)
print(len(t.split()))
