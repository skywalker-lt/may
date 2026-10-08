# AGENTS.md

Working rules for humans and agents contributing to `may`. Read `README.md` first for the state of knowledge; this
file is about how work is done and judged.

## What this repository is for

Routed YOLO constructs that have passed their receipts, the scripts that produce every published number, and the
records (tables, logs, seminar syntheses) that back them. It is not a scratch space: local tools, half-finished
experiments and pod-specific launchers stay out of version control until they are named deliverables.

## Non-negotiable rules

- **Routing is the subject.** A dense architecture, a better recipe or an engineering splice of public checkpoints
  is not a result here, however well it scores.
- **Devices of record.** T4 for latency-competing designs, L4 for accuracy-first designs; never mix devices in one
  comparison; label every number with its device, its runtime version and its protocol.
- **Protocol.** TensorRT fp16, batch 1, static engines, native median of 500 iterations with the comparator re-timed
  in the same session, per-layer profile kept; multi-label pycocotools on full val2017 from the CLI's dumps
  (conf 0.001, IoU 0.7, max_det 300). Every number carries the file it came from.
- **Bars.** The dense front at the design's own average latency + 0.003, and the share-matched null + 0.003. State
  the worst-case latency beside the average. Predictions are labelled "pred.", estimates "est.".
- **Training unit.** 80 epochs from the public Objects365 checkpoints with a dense control trained the same way,
  checkpoints backed up to the shared volume every 10 epochs, resumable launchers, the dataset cached to local disk.
  Short fine-tunes of converged public weights are not evidence (they drop before they recover).
- **Kill rules before launch.** Every run has its gate written down before it starts. A miss is a reported negative,
  not a reason to re-run.
- **Vocabulary.** A run of experiments is a "series", a "set" or a "scan"; the usual alternative word must not appear
  in files or messages.
- **Framing.** No surveillance or aerial tiny-object framing.
- **Git.** Deliver code only through git; small commits with a one-line subject and a body that states what was
  measured or changed; no co-author trailers; never commit datasets, checkpoints, engines, caches or local tools.
- **Documents** are written in English and delivered to the shared archive, not pasted into the README.

## Task routing

| task | where it lives | check before you claim it is done |
|---|---|---|
| a new routed construct | `may/<construct>/` once it has a passing receipt; before that, the fork's `dev/ds-yolo` branch | headroom bound on the dumps; engine builds on the device of record; timing with the comparator in-session |
| a measurement script | `scripts/` | reproduces a known row (the public YOLO26-M row) within 0.0005 AP / 0.02 ms before it measures anything new |
| a training launcher | `scripts/` | resumes from the last checkpoint after a kill; writes the backup every 10 epochs; caches to local disk |
| a table or figure | `records/` | every cell traceable to a log in `records/` |
| a seminar record | `records/seminars/` | synthesis plus full transcript; the vocabulary rule applied |

## Verification

- `ruff check` and `ruff format --check` on every changed Python file (line length 120, Google docstrings).
- Focused tests for the touched area; new public functions carry a runnable doctest.
- Any change to a number in `README.md` or `records/` names the log that produced it.

## Things already closed (do not re-propose without a new mechanism)

Static weight banks and residual deltas on the T4; soft per-pixel mixtures and scatter-based token top-k on either
device; crop-based local re-resolution; cascades; scene-routed label offsets; the dense M/L splice as a result;
the "chord room" between dense scales as headroom. See `README.md` for the measurements behind each.
