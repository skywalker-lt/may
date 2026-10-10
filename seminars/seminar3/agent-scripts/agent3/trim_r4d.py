import pathlib
p = pathlib.Path("/data/tmp/ds-yolo/seminar3/round4/agent3.md")
t = p.read_text()
R = [
("a natural-size arm stays in gate 1.", "a natural-size arm stays in step 1."),
("Not captured: whether the functions differ usefully; both\nreleased routers realised 0 bits after training.",
 "Not captured: whether the functions differ usefully; both released routers realised 0 bits after training."),
("0. Now, unconditionally: one dense 600-epoch control; two 80-epoch upcycled dense seeds (for the\n   seed spread);",
 "**Steps and gates.**\n\n0. Now, unconditionally: one dense 600-epoch control; two 80-epoch upcycled dense seeds (for the seed spread);"),
("and with a lowest-loss oracle. With s the paired spread\n   of (a):", "and with a lowest-loss oracle. With s the paired spread of (a):"),
("on both seeds. Otherwise E, with the pair\n   reported as the negative.", "on both seeds. Otherwise E, with\n   the pair reported as the negative."),
("whatever routing does. I would run step 0 (useful under every outcome) and step 1 (a few fine-tunes of 7M parameters), expect to stop,\nand treat a pass at step 1 as a surprise that has earned the next gate.",
 "whatever routing does. I would run step 0 (useful under every\noutcome) and step 1 (a few fine-tunes of 7M parameters), expect to stop, and treat a pass at step 1 as a surprise\nthat has earned the next gate."),
]
for a, b in R:
    assert a in t, a[:50]
    t = t.replace(a, b)
p.write_text(t)
print(len(t.split()), "scan" in t.lower())
