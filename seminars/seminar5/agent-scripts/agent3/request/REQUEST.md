# Agent 3 (direction 3, Mixture of Pyramid Levels): round-2 request

Priority: part 1 is a dump job of minutes and I ask for it before round 3. Part 2 is a training job, which I have cut from 3.5 to
about 2.5 H200 h (est.) and put below agents 7 and 1 in priority: round 2 showed that the level rungs are at best a +0.001
modifier of the exchange design (my `round2/agent3.md`, section 4).

Everything is in this directory: `run_r0.sh` (pod launcher), `build_branches.py`, `train_branch.py`, `onnx_build.py`,
`yolo26m-p45.yaml`, `yolo26m-p2leaf.yaml`, the built `b_init.pt` / `c_init.pt` and three ONNX graphs with sidecars. I
smoke-tested each piece on CPU with one thread (logs below).

## Part 1: three dumps of CPU-built static ONNX graphs (no training, about 0.2 H200 h)

`bash run_r0.sh dumps` runs the CLI dump step of `scripts/ds_yolo/dump_batch.sh` on each graph: TRT fp16, conf 0.001,
IoU 0.7, multi-label, full val2017, pycocotools.

| name | graph | what it measures |
|---|---|---|
| `m640_noP3head_o2o` | public YOLO26-M; the 6,400 P3 anchors are removed before the shipped end2end TopK; output [1,300,6] | what the shipped P4/P5 one-to-one heads keep without P3 (exact, untrained) |
| `m640_noP3head_o2m` | public YOLO26-M, one-to-many head, P3 anchors removed; raw [1,84,2000] for the CLI's NMS (`end2end: false` in the sidecar) | the same with the head that was never trained to suppress P3's objects |
| `m640_bconst_o2o` | `b_init.pt`: layers 14-17 removed, layer 17 = its train2017 per-channel mean (200 images), folded exactly into 19.cv1's BN mean | branch B at epoch 0 |

Checks (CPU, onnxruntime, one thread):
- On a real val image, the o2o graphs match PyTorch to 1.8e-4 over all rows scoring above 0.01. Random inputs only permute
  tied TopK rows.
- The o2m graph matches PyTorch to 2.7e-3.
- The B fold matches "public model with layer 17 replaced by the constant" to 3e-6 at P4 and P5.

## Part 2: B and its recipe control on the frozen public trunk (about 2.5 H200 h for both, est.)

`bash run_r0.sh train`, then `bash run_r0.sh dumps-trained`.

| run | init | frozen | trains | epochs |
|---|---|---|---|---|
| `b10` | `b_init.pt` (`yolo26m-p45.yaml`) | layers 0-13 (weights and BN statistics) | P4 out, P5 path, the P4/P5 one-to-many and one-to-one heads (public 19-23 renumbered 14-18; 7.95 M parameters) | 10 |
| `astar10` | public `yolo26m.pt` | layers 0-17 | the same layers 18-23, with P3 kept | 10 |

- **Data and recipe.** train2017 with `cache=disk`, 640, batch 64, the fork's stage-2 recipe (`recipe_yolo26m_stage2.yaml`,
  weight decay scaled to batch 64). Fine-tune settings as in the seminar-4 receipts: `warmup_epochs=0`,
  `warmup_bias_lr=0.00038`, `close_mosaic=3`. Losses: the trainer's own end2end loss (one-to-many plus one-to-one, TAL)
  over the live levels only. B's assigner sees strides 16 and 32 (`Detect nl=2`, smoke-checked).
- **Why A\* exists.** Every fine-tune of a converged public model under this fork's recipe has dropped to about 0.495-0.50.
  A\* retrains the identical parameter set under the identical recipe with P3 kept. The level effect is then read as
  B against A\*, with the recipe term cancelled.
- **Why `train_branch.py`.** The stock trainer rebuilds the model from YAML, and a 2-level Detect would then get P4-sized
  head widths, so the public head weights would be silently dropped. `BranchTrainer.get_model` returns the checkpoint's own
  module instead, both fresh and on resume.
- **Backups.** Every 5 epochs to `/training_data/runs_agent3/<run>/`, with resume from the local `last.pt` or the backup
  (the `train_80ep.py` rules).
- **Smoke test.** CPU, one thread: 1 epoch of `b10` on 8 images at 320. It trained 7,951,184 parameters with layers 0-13
  frozen, `nl=2`, strides [16, 32], ran validation, and saved a `last.pt` that keeps the public head widths.

**Forced dumps.** `dumps-trained` runs `dump_batch.sh` with `b10_last_640` and `astar10_last_640`, both o2o. The public
A is already dumped. I mix them on the CPU with the m640 router (`work/agent3/mix.py`, exact `evalImgs` recombination).

**Gates, written before the run.** All at share 0.5 with the m640 small-count router, T4 unset basis, B saving 1.00 ms (est.):
- **G1:** the mixture of B and A\*, minus A\*, must be at least -0.0011 (the optimistic proxy's value, tau16).
  - Below the untrained one-to-many floor (-0.0034 on my 1,000-image subset, round 2 section 1b), direction 3 closes.
    The pessimistic proxy, tau32, is -0.0083.
  - In between, the level rung is a modifier worth less than +0.001 and is reported in E only.
- **G2:** forced B's AP_M and AP_L must be within 0.004 of A\*. This is the P4/P5 loss of P3's bottom-up input, which no
  proxy models.
- **G3:** A\* minus public A gives the recipe term for head-only fine-tunes, a datum every direction can use.

The C branch (P2 leaf, `c_init.pt`, `--freeze 23`) is built and checked: layers 0-22 are bit-identical to the public model.
I withdraw its training. On the same router and footing, YOLO26-L beats the C proxy as the expensive rung (round 2,
section 4). A P2 refinement belongs in agent 5's tile form.
