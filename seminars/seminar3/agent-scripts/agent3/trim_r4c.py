import pathlib
p = pathlib.Path("/data/tmp/ds-yolo/seminar3/round4/agent3.md")
t = p.read_text()
R = [
("**Steps and gates.**\n\n", ""),
("(agent 1 is right)", "(agent 1)"),
("Round-3 consensus where the four\nfiles agree, my values otherwise.", "Round-3 consensus where it exists,\nelse my values."),
("I expect an attribution result at best.", "Expected: attribution at best."),
("Plainly: on the T4", "On the T4"),
("is one fine-tune and should not be dropped.", "is one fine-tune and should stay."),
("Now, under every outcome:", "Now, unconditionally:"),
]
for a, b in R:
    assert a in t, a[:50]
    t = t.replace(a, b)
p.write_text(t)
print(len(t.split()))
