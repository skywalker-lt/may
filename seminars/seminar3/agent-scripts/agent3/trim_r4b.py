import pathlib
p = pathlib.Path("/data/tmp/ds-yolo/seminar3/round4/agent3.md")
t = p.read_text()
def cut_block(t, start, end):
    i = t.index(start); j = t.index(end, i)
    return t[:i] + t[j:]
# drop the P(E) dissent (its content is in the V9 vote)
t = cut_block(t, "- **P(E) (V9).**", "- **Balanced shares (V3).**")
R = [
("below 0.01", "<0.01"),
("0 (one seed pair shows it with 0.08)", "0 (0.08 by chance)"),
("(the seed spread every\n   probability above assumes)", "(for the\n   seed spread)"),
("41 (45.2 in the `If`)", "41 (`If`: 45.2)"),
("5.359 (5.204 in the `If`)", "5.359 (`If`: 5.204)"),
("36 of 5,000, margins to 0.0090 / low / AP effect under 0.0005 est.; a 0.03 margin closes it", "36 of 5,000 / low / AP effect under 0.0005 est."),
("not measurable, scalar `If` / high for batched serving, none on the device of record", "not measurable, scalar `If` / high for batched serving"),
("calculated (`dq`), unm. / low with equal decay; delta-only decay removes 50-99.6% of an unsupported delta in 600 epochs", "calculated (`dq`), unm. / low with equal decay, high with delta-only decay at 600 epochs"),
("7.5% / high / 16.8M values on a quarter of the data", "7.5% / high / quarter data, no shared base"),
("Severities assume the rule of section 3; as coded\ntoday, collapse and train/deploy mismatch are high for the A rows (zero router gradient at top-1).", "Severities assume the rule of section 3; as coded,\nthe A rows have zero router gradient at top-1 (high)."),
("Between: one\n   repeat on a new seed, then stop if still between.", "Between: one\n   repeat, then stop."),
("unm. / medium / smallest branch gets 7.5% of calibration images", "unm. / medium / smallest branch 7.5% of calibration"),
("With base plus delta the base sees all 118,287 images. Not captured:", "Not captured:"),
("54 set flips / low / half the kernels change", "54 set flips / low"),
("unm. / low / trained as deployed", "unm. / low"),
("unm. / low / router fix mandatory", "unm. / low"),
("B, C and D are outside the cap, and the conditional family's 3% is a compile-path effect the dense\nmodel gets too.", "B, C and D are outside the cap; the conditional family's 3% is a compile-path effect the dense\nmodel shares."),
("Step 0 is useful under every outcome,\nand step 1 costs a few fine-tunes of 7M parameters. I would run those two, expect to stop, and treat a pass at\nstep 1 as a surprise that has earned the next gate.", "I would run step 0 (useful under every outcome) and step 1 (a few fine-tunes of 7M parameters), expect to stop,\nand treat a pass at step 1 as a surprise that has earned the next gate."),
]
for a, b in R:
    assert a in t, a[:50]
    t = t.replace(a, b)
i = t.index("- **Decay (V4).**"); j = t.index("- **Second 600-epoch dense seed (V7).**")
t = t[:i] + ("- **Decay (V4).** If the majority signs undecayed deltas, I dissent from my own round-3 text: the shrink in `dq`\n"
 "  (0.4-50% kept over 600 from-scratch epochs, calculated, not measured) then falls on the base alone, so the delta\n"
 "  grows relative to it and the model drifts toward A-full. Equal decay is neutral under the shared BatchNorm; the\n"
 "  logged norm ratio checks it.\n") + t[j:]
p.write_text(t)
print(len(t.split()))
